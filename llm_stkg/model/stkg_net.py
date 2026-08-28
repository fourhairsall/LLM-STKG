"""LLM-STKG 主干 (C2)：时空-语义异构图神经网络。

四类边做差异化消息传播（geo / category / semantic / covisit），再融合为 POI 表征；
用户侧用 GRU 编码轨迹 + 时间嵌入，最终对候选 POI 打分。

关键实现点（避免过平滑、保证可训练）：
- 每个 POI 的「基础表征」= ID 嵌入 + 类别 + 地理 + 语义 投影，始终保留自身身份；
- GNN 传播结果以「残差」方式叠加到基础表征（保留区分度，梯度可回流到 ID 嵌入）；
- padding 位置（轨迹不足 seq_len）用掩码置零，避免 -1 索引污染。
"""
import numpy as np
import torch
import torch.nn as nn


class STKGNet(nn.Module):
    def __init__(self, cfg, num_pois, num_cats, cat_ids, sem_vecs, edge_index,
                 n_users=0, user_item_edge=None, pop_prior=None, cooc_matrix=None):
        super().__init__()
        self.cfg = cfg
        self.num_pois = num_pois
        self.n_users = n_users
        self._cooc_matrix = cooc_matrix
        sem_raw = torch.as_tensor(sem_vecs, dtype=torch.float)  # 兼容 list / tensor（MD5占位 32 / 真实 BGE 768）
        # ---------- C1 贡献拆分：语义嵌入在本模型中有【两条】独立作用路径 ----------
        #   (a) 节点特征：sem 向量直接拼进 _base_feat()，为每个 POI 提供外部文本先验；
        #   (b) 边构造 + SGCP 门控：语义余弦决定 semantic 边集合，并门控 covisit 协同边。
        # 只报告"加 BGE vs 不加 BGE"会把两条路径的收益混在一起，无法回答审稿人
        # "语义信号究竟从哪来"。故引入 sem_feat_mode 单独切换路径 (a)：
        #   bge        : 默认，用真实 BGE 向量作节点特征（完整 C1）
        #   none       : 节点特征中不含任何语义向量（仅保留语义边+SGCP，隔离路径 b）
        #   cat_onehot : 用类目 one-hot 替代 BGE 作节点特征——关键的「类目名泄漏」对照。
        #                本数据集 POI 文本由 "{cat_name} near {lat:.2f},{lng:.2f}" 合成，
        #                BGE 编码的信息上界≈类目名+粗坐标；若 cat_onehot 与 bge 表现相当，
        #                则 C1 增益主要来自类目先验而非真实世界语义知识（须如实披露）。
        # 无论哪种模式，SGCP 门控与语义边始终使用【原始 BGE 向量】(sem_ref)，
        # 保证单变量对照——两模式之差仅为"节点特征里放什么"。
        self.register_buffer("sem_ref", sem_raw.clone())
        sem_mode = str(getattr(cfg, "sem_feat_mode", "bge")).lower()
        self.sem_feat_mode = sem_mode
        if sem_mode == "none":
            sem_tensor = torch.zeros(num_pois, 0)          # 0 维：等价于从拼接中移除
        elif sem_mode == "cat_onehot":
            _ci = torch.as_tensor(cat_ids, dtype=torch.long)
            sem_tensor = torch.zeros(num_pois, num_cats).scatter_(1, _ci.unsqueeze(1), 1.0)
        elif sem_mode == "id":
            # 无 LLM 文本域的语义兜底：可学习随机嵌入（非 BGE），对齐跨域"w/o LLM-text"消融；
            # 语义边/SGCP 应在 id 模式下关闭（use_semantic_edges=False）。
            id_dim = int(getattr(cfg, "id_sem_dim", 64))
            self.sem_id_emb = nn.Parameter(torch.randn(num_pois, id_dim) * 0.1)
            sem_raw = self.sem_id_emb.detach().clone()
            sem_tensor = self.sem_id_emb
        else:
            sem_tensor = sem_raw
        sem_dim = sem_tensor.shape[1]   # 动态匹配实际语义维度
        # cat_dim 可能=0（无类目域用 use_category_edges=False + cfg.cat_dim=0 关闭类目通道）。
        # 此时 cat_emb 不存在，基础表征跳过类目通道，feat_dim 亦不计入 cat_dim。
        self._cat_dim = cfg.cat_dim if getattr(cfg, "cat_dim", 0) > 0 else 0
        self.feat_dim = cfg.poi_dim + self._cat_dim + cfg.geo_dim + sem_dim
        # 用户长期偏好模块（C4）：补 CF 碾压 ours 的缺口——用户跨会话回访/长期偏好。
        # 关闭条件：未启用 或 无用户数（消融 ours 基础版时使用 n_users=0）。
        # 注意：session-based 测试协议无 user_id，C4 点积在测试时退化为常数（排序无效）；
        # 真正把协同信号带入推理的是下方的 User-POI 双图高阶传播（C5，烘焙进 POI 表征）。
        if getattr(cfg, "use_user_pref", True) and n_users > 0:
            self.user_emb = nn.Embedding(n_users, cfg.hidden_dim)

        self.poi_id_emb = nn.Embedding(num_pois, cfg.poi_dim)   # POI 身份特征（梯度直接回流）
        if self._cat_dim > 0:
            self.cat_emb = nn.Embedding(num_cats, self._cat_dim)
            self.cat_ids = torch.tensor(cat_ids, dtype=torch.long)
        else:
            self.cat_emb = None
            self.cat_ids = torch.zeros(num_pois, dtype=torch.long)  # 占位（cat_dim=0 时不被使用）
        if sem_mode == "id":
            self.sem_emb = self.sem_id_emb
        elif sem_dim == 0:    # sem_feat_mode=none：空张量注册为 buffer，避免优化器收到零元素参数
            self.register_buffer("sem_emb", sem_tensor)
        else:
            self.sem_emb = nn.Parameter(sem_tensor)
        self.geo_emb = nn.Parameter(torch.randn(num_pois, cfg.geo_dim) * 0.1)

        self.base = nn.Linear(self.feat_dim, cfg.hidden_dim)    # 基础表征投影（保留身份）
        # ---------- C2 消融：异构传播 vs 同质传播 ----------
        # homo_gnn=False（默认，C2 完整版）：每类边一个专属线性 W_t，四路输出拼接后融合——
        #   模型可学到"语义邻居与共访邻居应被区别加权"。
        # homo_gnn=True（消融）：四类边合并为一张并集图，共享同一个 W，单路输出——
        #   除"类型专属变换 + 类型级融合"外其余（含 SGCP 门控、残差、边集合）完全一致，
        #   因此两者之差即为「异构性」的净贡献。
        self.homo_gnn = getattr(cfg, "homo_gnn", False)
        self.use_residual = getattr(cfg, "use_residual", True)
        if self.homo_gnn:
            self.W = nn.ModuleDict({"__union__": nn.Linear(cfg.hidden_dim, cfg.hidden_dim)})
            self.combine = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        else:
            self.W = nn.ModuleDict({t: nn.Linear(cfg.hidden_dim, cfg.hidden_dim) for t in edge_index})
            self.combine = nn.Linear(cfg.hidden_dim * len(edge_index), cfg.hidden_dim)
        self.skip = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)   # 残差：保留基础表征

        self.extra_layers = None
        if cfg.num_gnn_layers > 1:
            self.extra_layers = nn.ModuleList(
                [nn.Linear(cfg.hidden_dim, cfg.hidden_dim) for _ in range(cfg.num_gnn_layers - 1)]
            )

        # ---------- 语义门控协同传播 (SGCP)：C1 语义桥引入协同信号 ----------
        # 对 covisit 协同边施加门控：gate = sigmoid(scale * sim(src,dst) + bias)，
        # 其中 sim 为 C1 语义嵌入的逐边余弦相似度（固定语义依据，仅学 gate 参数），
        # 仅让"语义相关"的共现信号流动——区别于 LightGCN 在 user-item 二部图上
        # 均匀传播（纯行为 CF，无语义桥）。
        self.use_sgcp = getattr(cfg, "use_sgcp", False)
        if self.use_sgcp:
            self.sgcp_scale = nn.Parameter(torch.tensor(float(getattr(cfg, "sgcp_scale", 1.0))))
            self.sgcp_bias = nn.Parameter(torch.tensor(float(getattr(cfg, "sgcp_bias", 3.0))))

        # 边索引规范化为 [2, E]（row0=src, row1=dst）。
        # 【修复】此前无条件 .t()：kg_builder 已按 [2, E] 输出，再转置后变成 [E, 2]，
        # 使 _propagate 里的 `src, dst = ei[0], ei[1]` 只取到「前两条边」，
        # 导致异构图传播每类边实际只传播 2 条（covisit 仅 1 条）→ C2 近乎空转。
        def _norm_ei(e):
            if e is None:
                return torch.empty(2, 0, dtype=torch.long)
            t_ = torch.as_tensor(np.asarray(e), dtype=torch.long)
            if t_.ndim != 2 or t_.numel() == 0:
                return torch.empty(2, 0, dtype=torch.long)
            if t_.shape[0] != 2 and t_.shape[1] == 2:   # [E, 2] → [2, E]
                t_ = t_.t()
            return t_.contiguous()

        self.edge_index = {t: _norm_ei(e) for t, e in edge_index.items()}

        # 同质图消融：把四类边拼成一张并集图，covisit 边置于末尾连续段，
        # 便于在不 clone 大张量的前提下仍对协同边施加 SGCP 门控（隔离单一变量）。
        if self.homo_gnn:
            non_cov = [ei for t, ei in self.edge_index.items() if t != "covisit" and ei.numel()]
            cov = [ei for t, ei in self.edge_index.items() if t == "covisit" and ei.numel()]
            parts = non_cov + cov
            u = torch.cat(parts, dim=1) if parts else torch.empty(2, 0, dtype=torch.long)
            self.register_buffer("u_src", u[0].contiguous())
            self.register_buffer("u_dst", u[1].contiguous())
            self.n_noncov = int(sum(ei.size(1) for ei in non_cov))
            self.n_cov = int(sum(ei.size(1) for ei in cov))

        # ---------- User-POI 二部图高阶传播（C5）：对标 LightGCN 的强项 ----------
        # 在 POI 异质 KG 之外，叠加 user-item 二部图，用 LightGCN 风格（无变换、无非线性、
        # 仅对称归一化层间聚合）做 K 层传播，把"用户跨会话回访/长期协同偏好"烘焙进 POI 表征；
        # 测试时无需 user_id（session-based 协议），POI embedding 已含 CF 协同信号。
        self.ui_edge = None
        if (getattr(cfg, "use_ui_graph", True) and n_users > 0
                and user_item_edge is not None and user_item_edge.numel() > 0):
            self.lgcn_user_emb = nn.Embedding(n_users, cfg.hidden_dim)
            u = user_item_edge[0].long()
            p = user_item_edge[1].long() + n_users
            row = torch.cat([u, p])
            col = torch.cat([p, u])
            n_nodes = n_users + num_pois
            deg = torch.zeros(n_nodes).index_add_(0, row, torch.ones(row.size(0)))
            deg_inv = (deg + 1e-8).pow(-0.5)
            norm = deg_inv[row] * deg_inv[col]
            self.register_buffer("ui_row", row)
            self.register_buffer("ui_col", col)
            self.register_buffer("ui_norm", norm)
            self.ui_edge = True
            self.lgcn_layers = getattr(cfg, "lgcn_layers", 2)
        # C5 融合：KG 与 CF 双可学习门控残差相加（初始均=1.0，两路信号充分参与，
        # 模型自适应学权重）；LayerNorm 稳定。取代此前 tanh(fuse) 饱和崩塌，
        # 也取代 cf_gate=0.1 对 CF 贡献的过度压制（那会让 C5 测试时≈纯 KG）。
        self.kg_gate = nn.Parameter(torch.tensor(1.0))
        self.cf_gate = nn.Parameter(torch.tensor(1.0))
        self.out_ln = nn.LayerNorm(cfg.hidden_dim)
        # 根因调试：纯 CF 协同打分项（可选）。单独用 bipartite LightGCN 生成 item 协同嵌入，
        # 以可学习门控加到最终分数上（cf_gate*(cf_emb[cand]·session_h)），用于隔离"是否缺 CF 协同信号"。
        self.cf_score_gate = None
        if getattr(cfg, "use_cf_score", False) and self.ui_edge is not None:
            self.cf_score_gate = nn.Parameter(torch.tensor(getattr(cfg, "cf_gate_init", 0.1)))

        self.traj_gru = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.temporal_emb = nn.Embedding(24 * 7, cfg.hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim), nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        # 每 epoch 缓存的 POI 表征（与 batch 无关，eval 时复用加速全候选排序）。
        self._poi_final = None

        # ---------- 行为先验通道 + 上下文门控（C6）----------
        # 动机（实测，见论文 §5.2/§5.4）：本基准 75.7% 的测试目标已出现在该样本自己的
        # 历史里；一个零参数规则"按历史出现次数排序"即得 Recall@10=0.6275，高于本模型
        # 早期版本的 0.4755，也高于 LightGCN / BPR-MF。原因是把 POI 当作不透明 ID 的
        # 神经打分头**无法表达"该用户已经来过这里 14 次"**——这类计数先验不在其假设空间内。
        #
        # 因此把三类行为先验显式接入打分函数，并由**上下文门控**决定各自权重：
        #   f_cnt = log(1 + 该候选在本次历史中出现次数)      → 严格包含 History-Frequency 规则
        #   f_rec = 1 / (该候选最近一次出现距序列末尾的步数) → 严格包含 History-Recency 规则
        #   f_pop = log(1 + 训练集全局频次) / 归一化          → 严格包含 Popularity 规则
        # 最终分数 s = w_kg·s_kg + w_cnt·f_cnt + w_rec·f_rec + w_pop·f_pop，w = softplus(gate(·)) ≥ 0。
        #
        # 关键设计：w 由**会话上下文**产生（会话表征 + 3 个重复度统计量），而不是全局常数。
        # 这使模型能按"这次会话像在重访还是在探索"逐样本调节先验强度——单纯把 HF 分数
        # 线性相加做不到这一点。令 w=(0,1,0,0) 即退化为 HF，(0,0,1,0) 为 HR，(0,0,0,1) 为 Pop，
        # 故本打分函数在假设空间上严格包含这三个平凡基线，不可能系统性劣于它们。
        self.prior_mode = str(getattr(cfg, "gate_mode", "off")).lower()   # context|global|off
        chans = str(getattr(cfg, "prior_channels", "")).strip()
        self.prior_channels = [c for c in chans.split(",") if c] if chans else []
        self.use_prior = bool(self.prior_channels) and self.prior_mode != "off"
        # 关键消融：整条 KG 语义通道是否还需要。若 use_kg_channel=False 时性能不降，
        # 则本基准上 KG+LLM 的净贡献为零，必须如实报告（这正是本工作最该自查的问题）。
        self.use_kg_channel = bool(getattr(cfg, "use_kg_channel", True))
        # C2 净贡献对照：True=跳过全部异构消息传递，只保留 skip 投影（见 _graph）
        self.no_graph = bool(getattr(cfg, "no_graph", False))
        # ---------- C6 cooc 通道（共现强度先验；跨域把"地理邻近"重定义为"会话/类目共现"）----------
        # cooc_matrix[n,n]：行归一化后的会话内共现强度（见 generic_loaders._build_cooc_matrix）。
        # 控计算量→稀疏化为每 POI 的 top-k 共现邻居；通道特征 = 候选与"本样本历史"的最强共现强度。
        if self._cooc_matrix is not None:
            cm = np.asarray(self._cooc_matrix, dtype=np.float32)
            topk = int(getattr(cfg, "cooc_topk", 50))
            topk = min(topk, int(cm.shape[1]) - 1) if int(cm.shape[1]) > 1 else 0
            if topk > 0:
                order = np.argpartition(-cm, topk, axis=1)[:, :topk]
                vals = np.take_along_axis(cm, order, axis=1)
                self.register_buffer("cooc_idx", torch.tensor(order, dtype=torch.long))
                self.register_buffer("cooc_val", torch.tensor(vals, dtype=torch.float32))
            else:
                self.cooc_idx = None
                self.cooc_val = None
            # ---- cooc_agg="sum" 专用：ItemKNN 方向的稠密共现表 ----
            # 方向很关键。cooc 由 generic_loaders 按**行**归一化（cooc[i,j] 除以第 i 行的最大值），
            # 因此 cooc[c,h] 与 cooc[h,c] 并不相同：
            #   · cooc[c,h]（按候选行归一化，稀疏 top-k 路径所用）等于对候选自身的最强共现做
            #     除法，是一种过度的 IDF 折扣 —— 会系统性压制热门候选，而 Steam 的答案分布
            #     本就高度偏向热门（Popularity 单独就有 R@10=0.0725）。
            #   · cooc[h,c]（按历史物品行归一化）才是经典 ItemKNN：score(c)=Σ_{h∈hist} cooc[h,c]。
            # 实测（Steam，悲观并列，全 2498 测试样本）：
            #   候选行方向 + top-k 求和 → R@10=0.0364(k=50) / 0.0468(k=200)，调 k 补不回来；
            #   历史行方向（本路径，等价 ItemKNN） → 0.0809。差距 1.7~2.2 倍全部来自归一化方向。
            # 只在 cooc_agg="sum" 且候选空间不大时物化 [N,N] 稠密表（N=1500 约 9MB）；
            # 超过阈值则退回稀疏 top-k 求和，避免 Foursquare(N=4980, 99MB + 25GFLOP/batch) 爆掉。
            # 注意：register_buffer 不允许覆盖已存在的普通属性，因此这里必须"要么 register、
            # 要么赋 None"，不能先赋 None 再 register（会抛 KeyError: attribute already exists）。
            _use_dense = False
            if str(getattr(cfg, "cooc_agg", "max")).lower() == "sum":
                _nmax = int(getattr(cfg, "cooc_dense_max_n", 3000))
                _use_dense = int(cm.shape[0]) <= _nmax
            if _use_dense:
                self.register_buffer("cooc_dense",
                                     torch.tensor(cm, dtype=torch.float32))
            else:
                self.cooc_dense = None
        else:
            self.cooc_idx = None
            self.cooc_val = None
            self.cooc_dense = None
        if self.use_prior:
            pv = torch.zeros(num_pois) if pop_prior is None else \
                torch.as_tensor(pop_prior, dtype=torch.float32).clamp_min(0)
            pv = torch.log1p(pv)
            pv = pv / (pv.max() + 1e-8)
            self.register_buffer("pop_feat", pv)                  # [N] ∈ [0,1]
            n_ch = (1 if self.use_kg_channel else 0) + len(self.prior_channels)
            # 上下文特征：3 个会话重复度统计量（见 forward 中 _ctx_feats）
            gate_in = cfg.hidden_dim + 3 if self.prior_mode == "context" else 0
            if self.prior_mode == "context":
                self.prior_gate = nn.Linear(gate_in, n_ch)
                nn.init.zeros_(self.prior_gate.weight)            # 起点=常数门控，再学上下文依赖
                # softplus(0.5413)=1.0：各通道初始权重均为 1，不预设偏好
                nn.init.constant_(self.prior_gate.bias, 0.5413)
            else:                                              # global：可学习但与上下文无关
                self.prior_w = nn.Parameter(torch.full((n_ch,), 0.5413))

    def to(self, *args, **kwargs):
        """覆盖 nn.Module.to：除参数/buffer 外，额外把『非注册辅助张量』移到同设备。

        GPU 正确性修复：__init__ 中 cat_ids 与 edge_index 是普通属性（非 Parameter/Buffer），
        父类的 to() 不会移动它们；在 GPU 下 cat_emb(cat_ids) 与 _propagate 用 edge_index 索引
        GPU 张量就会 device mismatch。此处随模型一并迁移。
        """
        super().to(*args, **kwargs)
        dev = None
        for b in self.buffers():
            dev = b.device
            break
        if dev is None:
            try:
                dev = next(self.parameters()).device
            except StopIteration:
                dev = torch.device("cpu")
        if getattr(self, "cat_ids", None) is not None and self.cat_ids.device != dev:
            self.cat_ids = self.cat_ids.to(dev)
        if getattr(self, "edge_index", None) is not None:
            for _t in self.edge_index:
                if self.edge_index[_t].device != dev:
                    self.edge_index[_t] = self.edge_index[_t].to(dev)
        return self

    def _base_feat(self):
        parts = [self.poi_id_emb.weight]
        if self.cat_emb is not None:
            parts.append(self.cat_emb(self.cat_ids))
        parts.append(self.geo_emb)
        parts.append(self.sem_emb)
        raw = torch.cat(parts, -1)
        return self.base(raw)                                   # [N, hidden]

    def _propagate_homo(self, x, Wdict):
        """C2 消融：并集图 + 共享 W 的同质传播（仍对 covisit 段施加 SGCP 门控）。"""
        N = x.size(0)
        dev = x.device                                          # GPU 修复：随输入设备建张量
        if self.u_src.numel() == 0:
            return torch.zeros(N, self.cfg.hidden_dim, device=dev)
        msg = Wdict["__union__"](x)
        src_msg = msg[self.u_src]
        if self.use_sgcp and self.n_cov > 0:
            k = self.n_noncov
            # 门控恒用原始 BGE 向量 sem_ref（不随 sem_feat_mode 变），保证消融单变量
            s_norm = self.sem_ref / (self.sem_ref.norm(dim=1, keepdim=True) + 1e-8)
            sim = (s_norm[self.u_src[k:]] * s_norm[self.u_dst[k:]]).sum(dim=1)
            gate = torch.sigmoid(self.sgcp_scale * sim + self.sgcp_bias)
            src_msg = torch.cat([src_msg[:k], src_msg[k:] * gate.unsqueeze(-1)], dim=0)
        agg = torch.zeros(N, self.cfg.hidden_dim, device=dev)
        agg.index_add_(0, self.u_dst, src_msg)
        deg = torch.zeros(N, device=dev).index_add_(
            0, self.u_dst, torch.ones(self.u_src.size(0), device=dev)).clamp_min(1.0)
        return agg / deg.unsqueeze(-1)

    def _propagate(self, x, Wdict):
        if self.homo_gnn:
            return self._propagate_homo(x, Wdict)
        outs, N = [], x.size(0)
        dev = x.device                                          # GPU 修复：随输入设备建张量
        for t, ei in self.edge_index.items():
            if ei.numel() == 0:
                outs.append(torch.zeros(N, self.cfg.hidden_dim, device=dev))
                continue
            src, dst = ei[0], ei[1]
            msg = Wdict[t](x)
            hi = msg[src]                                       # [E, hidden] 源消息（聚合值）
            if t == "covisit" and self.use_sgcp:
                # 门控恒用原始 BGE 向量 sem_ref（不随 sem_feat_mode 变），保证消融单变量
                s_norm = self.sem_ref / (self.sem_ref.norm(dim=1, keepdim=True) + 1e-8)
                sim = (s_norm[src] * s_norm[dst]).sum(dim=1)    # [E] 逐边语义余弦
                gate = torch.sigmoid(self.sgcp_scale * sim + self.sgcp_bias)
                hi = hi * gate.unsqueeze(-1)                    # SGCP：语义门控协同传播（作用于源消息）
            agg = torch.zeros(N, self.cfg.hidden_dim, device=dev)
            agg.index_add_(0, dst, hi)
            deg = torch.zeros(N, device=dev).index_add_(
                0, dst, torch.ones(src.size(0), device=dev)).clamp_min(1.0)
            outs.append(agg / deg.unsqueeze(-1))
        return torch.cat(outs, -1)

    def _graph(self, base_feat):
        # no_graph=True（消融）：完全不做消息传递，只保留 skip 线性投影。
        # 这是"异构图传播 C2 到底有没有净贡献"的干净对照——节点特征编码器（BGE 语义
        # + 类目 + 地理 + ID）、隐层维度、打分头、训练目标全部不变，唯一变量是有无传播。
        # 之所以必须做：修复 [2,E] 边索引 bug（此前图近乎空转）后 ours 指标不升反降，
        # 提示"真正参与传播的图"可能是负贡献，这个假设必须被正面检验而不是回避。
        if getattr(self, "no_graph", False):
            return self.skip(base_feat)
        # use_residual=False（消融）：去掉 skip(base_feat) 与层间恒等项，
        # 仅保留纯传播链路，用于验证"无残差 → POI 嵌入梯度消失、表征方差塌缩"。
        prop = self.combine(self._propagate(base_feat, self.W))
        if self.use_residual:
            prop = prop + self.skip(base_feat)
        h = prop
        if self.extra_layers is not None:
            for _ in self.extra_layers:
                p = self.combine(self._propagate(h, self.W))
                h = (p + self.skip(base_feat) + h) if self.use_residual else p
        return h

    def _lgcn(self, poi_h_kg):
        """User-POI 二部图 LightGCN 风格高阶传播（C5，重构版）。

        节点 = [users(0..n_users-1); POIs(n_users..n_users+num_pois-1)]，
        POI 初始化**直接用 KG 表征 poi_h_kg**（而非随机 cf_poi_init），保证 CF 协同信号
        与 KG 表征处于同一向量空间，相加即「协同细化」而非「噪声叠加」（修复此前
        R@10 从 0.257 暴跌到 0.0028 的空间错位问题）；User 初始化用可学习 lgcn_user_emb；
        对称归一化层间聚合 K 次，各层均值作为最终 embedding。返回的 POI 部分已烘焙
        「用户-物品协同偏好」，以残差门控方式加到 KG 表征上。
        """
        if self.ui_edge is None:
            return poi_h_kg
        n_nodes = self.n_users + self.num_pois
        E = torch.cat([self.lgcn_user_emb.weight, poi_h_kg], dim=0)  # [n_nodes, hidden]
        embs = [E]
        for _ in range(self.lgcn_layers):
            msg = E[self.ui_col] * self.ui_norm.unsqueeze(-1)       # 对称归一化消息
            agg = torch.zeros(n_nodes, self.cfg.hidden_dim, device=E.device)
            agg.index_add_(0, self.ui_row, msg)
            E = agg
            embs.append(E)
        Ef = torch.stack(embs, dim=0).mean(0)                       # 各层均值（LightGCN）
        return Ef[self.n_users:]                                    # 仅取 POI 部分

    # ------------------------------------------------------------------
    # v2：POI 表征与 batch 无关，但 forward 每调用一次都重算全图 GNN。
    # 训练循环在每 epoch 开头调用 refresh_poi_repr() 清空缓存，首个 batch 算出后
    # 复用至该 epoch 结束（GNN 参数在 epoch 内近似不变，等同 LightGCN 标准训练范式），
    # 从而把全图传播从每 epoch 算 B 次降为 1 次。
    def _poi_repr(self):
        base_feat = self._base_feat()                          # [N, hidden]
        poi_h_kg = self._graph(base_feat)                      # [N, hidden]（KG 传播，含残差）
        poi_cf = self._lgcn(poi_h_kg)                          # [N, hidden]（C5 协同传播）
        poi_final = self.out_ln(self.kg_gate * poi_h_kg + self.cf_gate * poi_cf)
        if getattr(self.cfg, "repr_center", False):
            poi_final = poi_final - poi_final.mean(dim=0, keepdim=True)
        return poi_final

    def _get_poi_repr(self):
        # 关键正确性：带梯度的 POI 表征不能跨 batch 缓存重用（第二次 backward 会报
        # "backward through the graph a second time"）。GNN 标准训练范式即每 batch 重算
        # 图传播（图很轻），故【训练时始终重算】以保证 item 嵌入参数收到正确梯度；
        # 仅在【eval/推理】时缓存（无需反向），加速全候选排序的重复调用。
        if not self.training and getattr(self.cfg, "cache_poi", True) and self._poi_final is not None:
            return self._poi_final
        pf = self._poi_repr()
        if not self.training and getattr(self.cfg, "cache_poi", True):
            self._poi_final = pf.detach()   # eval 无需梯度，detach 省显存
        return pf

    def refresh_poi_repr(self):
        """每 epoch 开头调用：清空 eval 缓存（训练本就不缓存，此处为保险）。"""
        self._poi_final = None

    def forward(self, traj_poi_ids, traj_time_bins, candidate_ids, user_ids=None):
        poi_final = self._get_poi_repr()                       # [N, hidden]（每 epoch 缓存）

        # 轨迹编码（padding 掩码：-1 → 0 且贡献置零）——本会话序列/融合表征
        mask = (traj_poi_ids >= 0).unsqueeze(-1).float()
        clamped = traj_poi_ids.clamp_min(0)
        tp = (poi_final[clamped] + self.temporal_emb(traj_time_bins)) * mask
        _sp = str(getattr(self.cfg, "session_pool", "gru")).lower()
        if _sp == "mean":
            denom = mask.sum(1).clamp_min(1.0)                 # [B,1]
            user_h = (tp * mask).sum(1) / denom                # 均值池化（CF 式用户表征）
        else:
            _, h_n = self.traj_gru(tp)
            user_h = h_n.squeeze(0)                            # [B, hidden]（GRU session 表征）

        cand_h = poi_final[candidate_ids]                      # [B, K, hidden]
        user_exp = user_h.unsqueeze(1).expand_as(cand_h)
        if getattr(self.cfg, "scorer", "mlp") == "dot":
            # CF 式点积打分（对标 BPR/LightGCN 的 u·v），隔离 MLP 打分器是否为瓶颈
            session_score = (cand_h * user_exp).sum(-1)        # [B, K]
        else:
            session_score = self.scorer(torch.cat([user_exp, cand_h], -1)).squeeze(-1)  # [B, K]

        # 根因调试：纯 CF 协同打分项（可选，正交叠加，不进 poi_final）
        if self.cf_score_gate is not None:
            cf_emb = poi_cf[candidate_ids]                     # [B, K, hidden]（bipartite 协同表征）
            cf_term = (cf_emb * user_h.unsqueeze(1)).sum(-1)   # [B, K]（统一用会话池化表征）
            session_score = session_score + self.cf_score_gate * cf_term

        # 用户长期偏好项（C4）：u_vec·cand_h。注：session-based 测试无 user_id，此项在
        # 测试时退化为常数（对排序无贡献）；其训练监督仍有正则作用，且与 C5 互补。
        if user_ids is not None and hasattr(self, "user_emb"):
            u_vec = self.user_emb(user_ids)                    # [B, hidden]
            pers = (u_vec.unsqueeze(1) * cand_h).sum(-1)       # [B, K]
            session_score = session_score + pers

        # ---------- C6：行为先验通道 + 上下文门控 ----------
        if self.use_prior:
            session_score = self._apply_prior(
                session_score, traj_poi_ids, candidate_ids, user_h)
        return session_score

    # ------------------------------------------------------------------
    def _hist_stats(self, traj_poi_ids):
        """由填充后的历史序列算出「每个候选的历史计数 / 近因权重」全 POI 表。

        traj_poi_ids : [B, T]，padding 用 -1。有效位置在前、padding 在后（见 _collate）。
        返回 cnt[B, N]（出现次数）、rec[B, N]（最近一次出现的近因权重 1/距末尾步数）。
        """
        B, T = traj_poi_ids.shape
        N = self.num_pois
        valid = (traj_poi_ids >= 0)
        clamped = traj_poi_ids.clamp_min(0)
        cnt = torch.zeros(B, N, device=traj_poi_ids.device)
        cnt.scatter_add_(1, clamped, valid.float())
        # 近因：位置 j（0-based）距序列末尾的步数 = L_b - j，权重取其倒数，越近越大（末位=1）
        L = valid.sum(1, keepdim=True).float().clamp_min(1.0)              # [B,1]
        pos = torch.arange(T, device=traj_poi_ids.device).unsqueeze(0).float()
        w = torch.where(valid, 1.0 / (L - pos).clamp_min(1.0),
                        torch.zeros_like(pos).expand(B, T))
        rec = torch.zeros(B, N, device=traj_poi_ids.device)
        rec.scatter_reduce_(1, clamped, w, reduce="amax", include_self=True)
        return cnt, rec

    def _ctx_feats(self, cnt, traj_poi_ids):
        """会话重复度上下文：门控据此判断本次会话偏"重访"还是"探索"。

        1) log(1+历史长度)/10        —— 长历史更可能重访
        2) 去重比 = #distinct/#total —— 越低越重复
        3) 最高频占比 = max_cnt/#total —— 头部 POI 的支配程度
        """
        tot = cnt.sum(1, keepdim=True).clamp_min(1.0)                      # [B,1]
        distinct = (cnt > 0).float().sum(1, keepdim=True)
        return torch.cat([torch.log1p(tot) / 10.0, distinct / tot,
                          cnt.max(1, keepdim=True).values / tot], dim=1)   # [B,3]

    def _apply_prior(self, session_score, traj_poi_ids, candidate_ids, user_h):
        cnt, rec = self._hist_stats(traj_poi_ids)
        # 通道 0 = KG 语义打分（消融时整条移除，而非置零权重——避免"权重虽小但仍参与"的歧义）
        feats = [session_score] if self.use_kg_channel else []
        for ch in self.prior_channels:
            if ch == "cnt":
                feats.append(torch.log1p(cnt.gather(1, candidate_ids)))
            elif ch == "rec":
                feats.append(rec.gather(1, candidate_ids))
            elif ch == "pop":
                feats.append(self.pop_feat[candidate_ids])
            elif ch == "cooc":
                # 候选与"本样本历史"的共现强度：取 cooc[c, ·]（行归一化∈[0,1]）的 top-k 邻居，
                # 与本样本历史求交后聚合。稀疏化避免 [B,N]@[N,N] 全矩阵乘法。
                #
                # ⚠️ 聚合方式必须随域切换（cooc_agg）——这是跨域实测出来的关键设计点：
                #   max（默认）：只取单个最强共现邻居。适合**短会话**（Foursquare trajectory
                #       模式历史均长 7.8）：证据本就稀少，取最强即可，且对噪声共现稳健。
                #   sum：对落在历史中的全部 top-k 邻居累加。适合**长而稠密的历史**
                #       （Steam 测试端均长 74.5、Gowalla 58.4）：此时几乎每个候选都能找到
                #       至少一个强共现邻居，max 迅速饱和到 ≈1.0，大量候选并列而丧失区分度；
                #       只有累加才能体现"有多少历史证据指向该候选"。
                # 实证（Steam，悲观并列，2498 测试样本，通道单独作打分器）：
                #   max（候选行 + top-k）    R@10=0.0072，tie_ratio=0.6148 —— 比 Random(0.0212)
                #                            还差：61% 候选并列榜首，通道等于噪声。
                #   sum（候选行 + top-k）    R@10=0.0364(k=50) / 0.0468(k=200)，tie_ratio=0.0119。
                #   sum（历史行 = ItemKNN）  R@10=0.0809 ← 本实现走这条稠密路径。
                # 对应到端到端：max 聚合下 ours 精确退化到热度水平（R@10=0.0729 vs Pop 0.0725），
                # 且换序列编码器（mean→GRU）毫无改善（0.0725）——因为跨域只剩 pop/cooc 两个
                # 通道，cooc 一失效就只剩 pop 可用。这是"有共现通道却没把共现用起来"的完整证据链。
                _agg = str(getattr(self.cfg, "cooc_agg", "max")).lower()
                if _agg == "sum" and getattr(self, "cooc_dense", None) is not None:
                    # ItemKNN 方向（历史行）：score(c) = Σ_{h∈hist} cooc[h, c]，一次稠密乘法。
                    hist_mask = (cnt > 0).float()                       # [B, N]
                    knn = hist_mask @ self.cooc_dense                    # [B, N]
                    cooc_feat = torch.log1p(knn.gather(1, candidate_ids))
                elif self.cooc_idx is not None:
                    hist_mask = (cnt > 0).float()                       # [B, N] 历史出现指示
                    cand_nb = self.cooc_idx[candidate_ids]               # [B, K, topk]
                    cand_nv = self.cooc_val[candidate_ids]              # [B, K, topk]
                    topk = cand_nb.shape[-1]
                    # 展开到 [B*K, N] 以便按候选逐个 gather 历史指示
                    hm_exp = hist_mask.repeat_interleave(cand_nb.shape[1], dim=0)
                    nb_in = hm_exp.gather(1, cand_nb.reshape(-1, topk)).reshape(
                        cand_nb.shape[0], cand_nb.shape[1], topk)
                    hit = cand_nv * nb_in.float()                        # [B, K, topk]
                    if str(getattr(self.cfg, "cooc_agg", "max")).lower() == "sum":
                        # log1p 压缩：命中数在长历史下可达数十，直接求和会让该通道量纲远超
                        # 其余通道（cnt 已是 log1p、rec/pop/kg 均 O(1)），非负门控只能整体
                        # 缩放而无法逐样本重标定，易压制其它通道。
                        cooc_feat = torch.log1p(hit.sum(dim=-1))        # [B, K]
                    else:
                        cooc_feat = hit.amax(dim=-1)                    # [B, K]
                else:
                    cooc_feat = torch.zeros_like(session_score)
                feats.append(cooc_feat)
        F_ = torch.stack(feats, dim=-1)                                    # [B, K, n_ch]
        if self.prior_mode == "context":
            g_in = torch.cat([user_h, self._ctx_feats(cnt, traj_poi_ids)], dim=1)
            w = torch.nn.functional.softplus(self.prior_gate(g_in))        # [B, n_ch] ≥ 0
        else:
            w = torch.nn.functional.softplus(self.prior_w).unsqueeze(0)    # [1, n_ch]
        self._last_gate_w = w.detach().mean(0)                             # 诊断：平均通道权重
        return (F_ * w.unsqueeze(1)).sum(-1)                               # [B, K]
