from dataclasses import dataclass


@dataclass
class Config:
    # ---- 数据 ----
    num_users: int = 200
    num_pois: int = 500
    num_categories: int = 10
    seq_len: int = 20
    hist_mode: str = "trajectory"  # trajectory=历史取当前会话内前缀（旧行为，训练历史均值仅 7.8）
                                   # user=同一用户各会话按时间拼接后滑窗（与官方测试协议一致，
                                   # 测试历史均值 143.2）。见 train._build_samples 的说明

    # ---- 模型维度 ----
    poi_dim: int = 64
    cat_dim: int = 16
    geo_dim: int = 16
    sem_dim: int = 32
    hidden_dim: int = 64
    num_gnn_layers: int = 2

    # ---- C2 消融开关（异构传播 / 残差）----
    homo_gnn: bool = False       # True=同质图消融：四类边合并为并集图、共享同一个 W（去掉类型专属变换与类型级融合），
                                 # 仅保留 SGCP 门控以隔离"异构性"这一单一变量；False=异构传播（C2 完整版）
    use_residual: bool = True    # True=保留 skip(base_feat) 残差（C2 防过平滑）；False=消融残差，用于验证
                                 # "无残差则 POI 嵌入梯度消失、表征方差塌缩"的论断

    # ---- 旅游知识图谱 (C1) ----
    geo_radius_km: float = 2.0        # 地理邻近边阈值
    semantic_topk: int = 5            # 语义边保留数（备用）
    semantic_sim_thr: float = 0.30    # LLM 文本关系相似度阈值（过高→无边，过低→全连噪声）
    covisit_min: int = 3              # 共访边最小共现次数
    # 共访边「选谁当邻居」的打分方式。raw=原始共现次数（旧行为）；
    # cosine=cooc/√(f_a·f_b)；pmi=log(cooc·N_sess/(f_a·f_b))。后两者去热度偏置。
    # 稠密消费域必须去偏，否则枢纽垄断 → GNN 表征塌缩（hub_collapse_probe.py 实测，
    # 每节点 top-10）：Steam raw 邻居 Jaccard=0.5527/枢纽覆盖 77.6%/邻居多样性仅 11.5%，
    # pmi 下变为 0.0295/14.7%/80.7%；MovieLens-1M raw 0.3160/60.7%/16.5% 同样病态；
    # Gowalla raw 已是 0.0396/18.5%/71.0%（健康），故其 GNN 训练正常。
    covisit_score: str = "raw"
    max_degree: int = 10              # 每类关系每 POI 保留的最大邻居数（k-NN 剪枝；0=不剪枝）。
                                      # 阈值(geo_radius_km/semantic_sim_thr)退化为候选集生成器，
                                      # 图密度由 k 控制，避免高密城区 geo 平均度数 400+ 造成过平滑
    use_bge: bool = False             # 是否用真实 BGE 嵌入激活 C1（取代 MD5 哈希占位）
    bge_model_dir: str = "bge_model"  # 本地 bge 模型目录（sentence-transformers 格式）

    # ---- 语义门控协同传播 (SGCP, C1 引导的协同信号) ----
    use_sgcp: bool = False            # 是否启用 SGCP：covisit 协同信号经 C1 语义相似度门控后再传播
    sgcp_scale: float = 1.0          # 门控缩放系数（可学习参数）
    sgcp_bias: float = 3.0           # 门控偏置（初始较大→gate≈1 中性起点；训练后压低语义无关的共现噪声边）

    # ---- 训练 ----
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    neg_samples: int = 10
    loss_type: str = "ce"          # 训练目标：'ce'(带负采样 softmax 分类=InfoNCE) | 'bpr'(成对排序损失，本架构已证不兼容→分数塌缩) | 'list'(ListNet 式 listwise：温度锐化 softmax 直接优化顶部排序/NDCG)
    list_tau: float = 0.5         # listwise/温度锐化系数：sc/tau 后算 softmax；tau<1 越锐→顶部排序越自信→NDCG 越高（CE 是 tau=1 的特例）
    bpr_negs: int = 0             # BPR 额外负样本数：0=复用 neg_samples 候选(默认)；>0=在训练时另采样更多负样本增强排序信号
    # ---- C1 语义 hard-negative mining（保留 CE 损失以维持全局校准）----
    hard_neg_ratio: float = 0.0  # 训练候选中"语义近邻负样本"占比(0~1)：0=纯随机负样本(默认)；>0=按该比例用 bge 语义近邻替换随机负样本
    hard_neg_topk: int = 50      # 每个 POI 取语义 top-k 近邻作为 hard 负样本候选池（实际取多少由 hard_neg_ratio 决定）
    hard_neg_cache: str = "poi_bge_emb.npy"  # 离线 BGE 语义嵌入缓存（构建近邻池用）
    seed: int = 42
    device: str = "cpu"
    use_user_pref: bool = True   # 是否启用用户长期偏好模块（补 CF 碾压 ours 的缺口：用户回访/长期偏好）
    use_ui_graph: bool = True    # 是否启用 User-POI 二部图 LightGCN 高阶传播（C5：对标 LightGCN 强项，将协同信号烘焙进 POI 表征）
    lgcn_layers: int = 2         # LightGCN 传播层数 K（无变换/无非线性，仅层间对称归一化聚合）

    # ---- 根因调试开关（隔离"为何 ours 比 LightGCN/BPR 低 3×"）----
    scorer: str = "mlp"          # 打分器：'mlp'(当前 concat+MLP) | 'dot'(session_h·cand_h，CF 式点积，对标 BPR/LightGCN)
    session_pool: str = "gru"    # 会话编码：'gru'(当前) | 'mean'(轨迹 POI 表征均值池化)
    use_cf_score: bool = False   # 是否加"纯 CF 协同打分项"：单独 bipartite LightGCN item 嵌入，加 cf_gate*(cf_emb[cand]·session_h)
    cf_gate_init: float = 0.1    # cf 项初始权重（小→先当增量，验证 CF 信号是否真有益）

    # ---- 跨数据集 / 无文本域 消融开关 ----
    id_sem_dim: int = 64            # Steam 等无文本域：id-mode 语义兜底特征维度（可学习，非 LLM）
    use_geo_edges: bool = True      # False=跳过地理邻近边（数据集无 geo 信息，如 MovieLens/Steam）
    use_category_edges: bool = True # False=跳过类目边（数据集无类目信息，如 Gowalla/Steam）
    use_semantic_edges: bool = True # False=跳过语义边（无 LLM 文本 / w/o LLM-text 跨域消融）
    use_covisit_edges: bool = True  # False=跳过纯行为共访边（P1-5 单边缘类型消融：covisit-only 对照）
    cooc_topk: int = 50             # C6 cooc 通道：每 POI 保留共现强度 top-k 邻居（稀疏化，控计算量）
    # C6 cooc 通道的聚合方式，随「历史长度」切换（详见 stkg_net._apply_prior 注释）：
    #   "max"：取单个最强共现邻居 —— 短会话（Foursquare traj 均长 7.8）默认，保持既有结果不变。
    #   "sum"：对命中历史的 top-k 邻居 log1p 累加 —— 长稠密历史（Steam 74.5 / Gowalla 58.4）必需，
    #          否则 max 饱和、候选大面积并列，模型退化为热度打分。
    cooc_agg: str = "max"
    # cooc_agg="sum" 时物化 [N,N] 稠密共现表（走 ItemKNN 的历史行方向）的候选空间上限。
    # N=1500 约 9MB / 2.3 GFLOP·batch，可忽略；N=4980（Foursquare）则为 99MB / 25 GFLOP·batch，
    # 超限自动退回稀疏 top-k 求和。Foursquare 本就用 max，不受影响。
    cooc_dense_max_n: int = 3000

    # ---- C6 行为先验通道 + 上下文门控 ----
    # 实测：本基准 75.7% 测试目标已在自身历史中，零参数的"历史频次"规则 R@10=0.6275，
    # 高于所有神经基线。把计数/近因/热度三类先验显式接入打分，并用会话上下文学门控权重，
    # 使打分函数在假设空间上严格包含 HF / HR / Pop 三个平凡基线。
    prior_channels: str = ""     # 逗号分隔子集，取值 cnt/rec/pop；空=不启用（保持旧行为）
    gate_mode: str = "off"       # context=按会话上下文逐样本产生权重 | global=全局可学习标量 | off=停用
    use_kg_channel: bool = True  # False=把整条 KG 语义打分通道从融合中移除（不是权重置零，是不参与 stack），
                                 # 用于回答审稿人"去掉 LLM/KG 后剩下的先验通道能否达到同样效果"这一致命消融
    no_graph: bool = False       # True=完全不做异构消息传递（只保留 skip 投影）。C2 传播净贡献的干净对照：
                                 # 特征编码器/维度/打分头/训练目标全不变，唯一变量是有无图传播

    # ---- v2 速度/效果增强 ----
    # 每 epoch 缓存 POI 表征：poi_final 与 batch 无关。但注意——带梯度的张量不能跨 batch
    # 缓存复用（会触发"backward through the graph a second time"），GNN 标准训练范式即每
    # batch 重算图传播（图很轻），故【训练时始终重算】以保证 item 嵌入参数收到正确梯度；
    # POI 表征缓存仅用于【eval/推理】（无需反向），加速全候选排序的重复调用；
    # 训练仍每 batch 重算图传播以保证 item 嵌入收到正确梯度。默认开启，eval 加速且无害。
    cache_poi: bool = True

    # ---- 跨节点中心化（公共均值消除）----
    # 诊断依据（Steam, 600 POI）：poi_repr_var_mean=8.85（差异分量并不小）却 pairwise_cos=0.9997，
    # 说明表征不是"没信息"，而是所有 POI 共享一个巨大公共向量 μ，差异 δ 被淹没。此时
    #     u·h_c = ‖μ‖² + μ·δ_c + δ_u·δ_c
    # 第二项（与用户无关的物品偏置）以 ‖μ‖/‖δ_u‖ 的比例碾压第三项（真正的个性化），且
    # δ_u 是 74 个历史项的均值、再自带 1/√74 衰减 → 点积事实上退化成全局物品偏置，
    # C6 门控随即把 KG 通道压到 0.0345、把权重全给 cooc 规则通道（实测 [0.0345, 0.0023, 6.3154]）。
    # 注意 out_ln 是 LayerNorm：它只在**单节点的维度方向**上中心化，对**跨节点公共分量 μ**
    # 完全无效——这正是漏掉的一刀。repr_center=True 时对 poi_final 做跨 POI 去均值，
    # 使打分严格等于 δ_u·δ_c。默认 False 以保护既有 Foursquare 结果（那里 cos=0.80，μ 不主导）。
    repr_center: bool = False

    def __str__(self):
        return "\n".join(f"  {k}={v}" for k, v in self.__dict__.items())
