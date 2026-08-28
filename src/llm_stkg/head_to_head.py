"""真数据头对头：在真实 Foursquare-NYC (LLM4POI) 上做公平对比。

流程：
  1. 加载真实数据（load_real_nyc → 连续索引后的 pois/checkins/test_samples）。
  2. 仅用训练集构建旅游知识图谱 (C1)。
  3. 训练 ours(LLM-STKG) 与 4 个基线（同协议、全候选排名）。
  4. 在**官方测试集**（test_pairs，1447 样本，与训练互斥）上评估 Recall@5/10、NDCG@5/10。
  5. 若提供 --sota_preds（SOTA 模型在本测试集上的全候选打分），一并算指标，得完整头对头表。

公平要点：
  - 所有模型在同一测试集、同一全候选排名协议下评估；
  - 基线改用 session_predict（历史物品均值表征），避免跨用户 test 无 user embedding 的不公平；
  - ours 本身即 session-based（不依赖 user_id）。

用法（在用户本机，torch 可用）：
  python -m llm_stkg.head_to_head --device cuda --epochs 30 --out head_to_head.json
  # 接入 SOTA 预测：
  python -m llm_stkg.head_to_head --device cuda --sota_preds sota_preds.json
"""
import argparse
import json
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .kg.kg_builder import TourismKG
from .kg.llm_interface import LLMInterface
from .model.stkg_net import STKGNet
from .data.foursquare_loader import load_real_nyc
from .data import generic_loaders as GL
from .baselines import build_baselines
from .evaluate import rank_metrics, rank_diag, mask_history
from .train import _build_samples, TrajDataset, _collate


# ---------- ours 训练（全量训练集，测试集单独评估）----------
def build_ui_edge(checkins, num_pois):
    """从训练 checkins 构建 User-POI 二部图边（用于 C5 双图高阶传播）。"""
    rows, cols = [], []
    for u, seq in checkins:
        for p in set(int(x) for x in seq):
            if 0 <= p < num_pois:
                rows.append(u)
                cols.append(p)
    if not rows:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long)


def build_hard_neg_pool(cfg, num_pois):
    """用离线 BGE 语义嵌入构建每个 POI 的语义近邻硬负样本池（C1 语义 hard-negative mining）。

    返回 np.ndarray [num_pois, hard_neg_topk]（int64），每行是 target 的语义 top-k 近邻 POI 索引（已排除自身）。
    - 仅当 use_bge 且缓存存在、且索引对齐(num_pois == 嵌入行数)时启用；
    - 否则返回 None（train_ours 中退回纯随机负样本），保证架构安全、不引入纯 CF。
    该池用于训练阶段把部分随机负样本替换为"语义近似近邻"，逼模型分离近似重复 POI，
    直接提升全候选排名顶部质量（pct_rank1 / NDCG），同时保留 CE 损失的全局 softmax 校准。
    """
    if getattr(cfg, "hard_neg_ratio", 0.0) <= 0:
        return None
    if not getattr(cfg, "use_bge", False):
        return None
    cache = getattr(cfg, "hard_neg_cache", "poi_bge_emb.npy")
    if not os.path.exists(cache):
        print(f"[hard-neg] 缓存 {cache} 不存在，退回随机负样本")
        return None
    emb = np.load(cache)
    if emb.shape[0] != num_pois:
        print(f"[hard-neg] 嵌入行数 {emb.shape[0]} != num_pois {num_pois}（索引已重映射），退回随机负样本")
        return None
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)  # L2 归一化
    sim = emb @ emb.T  # [N, N] 余弦相似度
    topk = int(getattr(cfg, "hard_neg_topk", 50))
    topk = min(topk, num_pois - 1)
    # 每行取 top-(topk+1) 近邻（含自身），去掉自身，留 topk
    order = np.argpartition(-sim, topk + 1, axis=1)[:, :topk + 1]
    pool = np.full((num_pois, topk), -1, dtype=np.int64)
    for i in range(num_pois):
        nbrs = [int(j) for j in order[i] if int(j) != i][:topk]
        if nbrs:
            pool[i, :len(nbrs)] = nbrs
    print(f"[hard-neg] 已构建语义近邻池: {num_pois}×{topk}（每 POI top-{topk} 语义近邻）")
    return pool


def build_kg(cfg, pois, checkins):
    """构建旅游知识图谱（训练与解释模块共用，保证图完全一致）。"""
    bge_dir = cfg.bge_model_dir if getattr(cfg, "use_bge", False) else None
    # C1 真实 BGE 嵌入：优先加载离线分块编码缓存（poi_bge_emb.npy），避免训练时一次性
    # 编码 4980 POI 触发沙箱 segfault；缓存不存在时退回 LLM 接口实时编码（仅小数据时安全）。
    sem_vecs = None
    if bge_dir is not None:
        cache = getattr(cfg, "bge_cache", "poi_bge_emb.npy")
        if os.path.exists(cache):
            sem_vecs = np.load(cache)
    return TourismKG(cfg, LLMInterface(bge_model_dir=bge_dir)).build(
        pois, checkins, sem_vecs=sem_vecs)


def build_pop_prior(checkins, num_pois):
    """训练集 POI 频次向量（C6 热度通道输入）。只用训练数据，无测试泄漏。"""
    from collections import Counter
    freq = Counter(p for _, seq in checkins for p in seq)
    v = torch.zeros(num_pois, dtype=torch.float32)
    for p, c in freq.items():
        if 0 <= p < num_pois:
            v[p] = float(c)
    return v


def train_ours(cfg, pois, checkins, train_samples, num_pois, device, n_users=0, user_item_edge=None,
               cooc_matrix=None):
    device = device or cfg.device
    kg = build_kg(cfg, pois, checkins)
    print("[KG] 边统计:", kg.stats())
    print(f"[C5] User-POI 双图: {'启用' if (user_item_edge is not None and user_item_edge.numel()>0) else '禁用'}")
    model = STKGNet(cfg, num_pois, kg.num_cats, kg.cat_ids, kg.sem_vecs, kg.edge_index,
                    n_users=n_users, user_item_edge=user_item_edge,
                    pop_prior=build_pop_prior(checkins, num_pois),
                    cooc_matrix=cooc_matrix).to(device)
    if getattr(model, "use_prior", False):
        print(f"[C6] 先验通道={model.prior_channels} 门控={model.prior_mode} "
              f"KG通道={'启用' if getattr(model, 'use_kg_channel', True) else '已移除(消融)'}")
    rng = random.Random(cfg.seed)
    hard_pool = build_hard_neg_pool(cfg, num_pois)
    ds = TrajDataset(train_samples, num_pois, cfg.neg_samples, rng,
                     hard_neg_pool=hard_pool, hard_neg_ratio=cfg.hard_neg_ratio)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    use_bpr = getattr(cfg, "loss_type", "ce").lower() == "bpr"
    use_list = getattr(cfg, "loss_type", "ce").lower() == "list"
    list_tau = float(getattr(cfg, "list_tau", 1.0))
    import time as _time
    _t0 = _time.time()
    _grad_norms = []          # 最后一个 epoch 内 poi_id_emb 的梯度范数（过平滑/梯度消失证据）
    def _compute_loss(sc, Y):
        if use_bpr:
            # 成对排序损失（BPR）：直接优化"正样本得分 > 各负样本得分"。
            # ⚠️ 已证对本架构（非共享嵌入点积打分头）不兼容：梯度在 10 个随机负样本被 beat 后消失，
            # 导致全候选分数塌缩（pct_rank1=0.62%、median_rank=1882）→ 全排名崩盘。保留仅供对照，不建议用于生产。
            pos = sc[:, 0]                       # [B] 正样本得分
            neg = sc[:, 1:]                      # [B, K-1] 负样本得分
            return -F.logsigmoid(pos.unsqueeze(1) - neg).mean()
        elif use_list:
            # ListNet 式 listwise 损失：理想相关性为 one-hot（仅目标相关），其排列概率 = softmax(y) = delta；
            # ListNet 损失退化成对预测分布的 -log 似然 = 在 sc/tau 上的交叉熵。tau<1 锐化 softmax，
            # 迫使模型把目标得分推到更绝对主导 → 直接优化顶部排序质量与 NDCG（CE 是 tau=1 的特例）。
            # 相比 BPR，listwise 保留全局 softmax 分母的对比信号，分数可在全空间分离，故不塌缩。
            return F.cross_entropy(sc / list_tau, Y)
        else:
            return loss_fn(sc, Y)
    for ep in range(cfg.epochs):
        model.train()
        if getattr(model, "refresh_poi_repr", None) is not None:
            model.refresh_poi_repr()   # 清空 eval 缓存
        total = 0.0
        _grad_norms = []
        for H, T, C, Y, U, _ in dl:
            H, T, C, Y, U = H.to(device), T.to(device), C.to(device), Y.to(device), U.to(device)
            sc = model(H, T, C, U)   # [B, 1(正)+neg_samples]；正样本恒在候选 index 0（见 TrajDataset 的 cands=[target]+negs, labels=0）
            loss = _compute_loss(sc, Y)
            opt.zero_grad(); loss.backward()
            g = model.poi_id_emb.weight.grad
            if g is not None:
                _grad_norms.append(float(g.norm().item()))
            opt.step()
            total += loss.item()
        print(f"[ours Epoch {ep+1:02d}] loss={total/len(dl):.4f}")
    # ---- 训练诊断（残差/过平滑证据 + 系统指标，供论文可复现小节引用）----
    model.eval()
    with torch.no_grad():
        _h = model._graph(model._base_feat())            # [N, hidden] KG 传播后 POI 表征
        _repr_var = float(_h.var(dim=0).mean().item())   # 跨 POI 的逐维方差均值：越小越接近塌缩
        _hn = _h / (_h.norm(dim=1, keepdim=True) + 1e-8)
        _idx = torch.randperm(_h.size(0))[:512]
        _pc = (_hn[_idx] @ _hn[_idx].t())
        _off = _pc[~torch.eye(_pc.size(0), dtype=torch.bool)]
        _mean_cos = float(_off.mean().item())            # 平均两两余弦：越接近 1 越过平滑
        # 公共分量 μ 对差异分量 δ 的碾压倍数 = ‖μ‖ / mean‖h−μ‖。
        # 这是比 pairwise_cos 更可解释的量：点积 u·h_c 中「与用户无关的物品偏置项 μ·δ_c」
        # 与「个性化项 δ_u·δ_c」的量级比正比于此值（δ_u 还自带 1/√|hist| 衰减）。
        _mu = _h.mean(dim=0, keepdim=True)
        _delta_norm = float((_h - _mu).norm(dim=1).mean().item())
        _mu_ratio = float(_mu.norm().item()) / max(_delta_norm, 1e-8)
    model._diag = {
        "poi_emb_grad_norm_mean_last_epoch": (sum(_grad_norms) / len(_grad_norms)) if _grad_norms else None,
        "poi_repr_var_mean": _repr_var,
        "poi_repr_pairwise_cos_mean": _mean_cos,
        "poi_repr_mu_over_delta": round(_mu_ratio, 3),   # ‖μ‖/mean‖δ‖：公共分量碾压倍数
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "n_trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "train_seconds": round(_time.time() - _t0, 1),
        "epochs": cfg.epochs,
        "homo_gnn": bool(getattr(cfg, "homo_gnn", False)),
        "no_graph": bool(getattr(cfg, "no_graph", False)),
        "repr_center": bool(getattr(cfg, "repr_center", False)),
        "cooc_agg": str(getattr(cfg, "cooc_agg", "max")),
        "covisit_score": str(getattr(cfg, "covisit_score", "raw")),
        "use_residual": bool(getattr(cfg, "use_residual", True)),
        "session_pool": getattr(cfg, "session_pool", "gru"),
        "scorer": getattr(cfg, "scorer", "mlp"),
        "sem_feat_mode": getattr(cfg, "sem_feat_mode", "bge"),
        # C6：训练结束时各通道的平均门控权重 [w_kg, w_cnt, w_rec, w_pop]（按启用顺序）。
        # 这是"模型到底依赖 KG 还是依赖历史计数"的直接证据，必须报告。
        "prior_channels": list(getattr(model, "prior_channels", [])),
        "gate_mode": getattr(model, "prior_mode", "off"),
        "use_kg_channel": bool(getattr(model, "use_kg_channel", True)),
        "gate_w_mean": ([round(float(x), 4) for x in model._last_gate_w.tolist()]
                        if hasattr(model, "_last_gate_w") else None),
        "semantic_sim_thr": getattr(cfg, "semantic_sim_thr", None),
        "max_degree": getattr(cfg, "max_degree", 0),
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "hist_mode": getattr(cfg, "hist_mode", "trajectory"),
        "seq_len": cfg.seq_len,
        "neg_samples": cfg.neg_samples,
        "loss_type": getattr(cfg, "loss_type", "ce"),
        "kg_edges": {t: int(ei.shape[1]) for t, ei in model.edge_index.items()},
    }
    print(f"[diag] {model._diag}")
    return model


def eval_session(model, test_samples, num_pois, device, batch=512, mask_hist=False):
    """用基线的 session_predict 在测试集上全候选排名。

    mask_hist: 见 evaluate.mask_history 的说明。无重复消费域（ml1m/steam/gowalla）须置 True，
               且必须与 ours 使用同一开关，否则对比不公平。
    """
    device = device or "cpu"
    if not test_samples:
        return {"Recall@5": 0.0, "NDCG@5": 0.0, "Recall@10": 0.0, "NDCG@10": 0.0}
    scores, tgts = [], []
    for u, hist, tgt in test_samples:
        sc = torch.tensor(model.session_predict(hist), dtype=torch.float32)
        scores.append(sc); tgts.append(tgt)
    scores = torch.stack(scores); tgts = torch.tensor(tgts, dtype=torch.long)
    if mask_hist:
        scores = mask_history(scores, test_samples, num_pois)
    m = rank_metrics(scores, tgts, k_list=(5, 10))
    m["__rank_diag__"] = rank_diag(scores, tgts)
    return m


def apply_dataset_cfg(cfg, name):
    """跨域数据集（SASRec 基准 ml1m/gowalla/steam）的默认配置。

    统一以 **w/o LLM-text** 模式评测：跨域物品无可用文本，故语义特征与语义边均关闭，
    仅保留行为先验 C6 + 共现 cooc 通道 +（按域可选的）地理/类目边。
    如此可隔离"LLM 语义之外"的通用序列推荐能力，是与 SASRec/eSASRec 的公平对照。

    ⚠️ 关键域适配：cnt / rec 两个通道在跨域**必须关闭**
    -----------------------------------------------------
    C6 的门控权重 w = softplus(·) ≥ 0，只能非负。在 Foursquare 这类重访主导域
    （测试端 revisit_ratio=0.7574）这没问题：cnt/rec 在正样本上普遍非零，正权重恰好表达
    「该用户来过这里 N 次」。但在 ml1m/steam/gowalla 这类**无重复消费**域（revisit≈0）：
      · 正样本的 f_cnt = log1p(0) = 0、f_rec = 0，**恒为零**，不含任何判别信息；
      · 只有恰好落在历史里的负样本才非零（Steam 上约 9% 的随机负样本）。
    于是这两个通道只能给「永远不是答案」的历史物品加分，而非负门控**无法学出负权重**
    去反转它——只能把 w 往 0 挤，梯度还随之衰减。实测后果：Steam 上训练 loss 从第 2 轮起
    死锁在 2.2100（neg=10 的 11 类 CE 随机基准 ln(11)=2.3979），模型退化为近似热度打分。
    因此跨域默认 prior_channels="pop,cooc"。这与评测端的 mask_history 是同一条领域事实
    （"在历史里 ⇒ 不是答案"）的训练侧与推理侧两个体现。
    """
    # 公共跨域默认
    cfg.hist_mode = "user"
    cfg.seq_len = 200
    cfg.use_bge = False
    cfg.use_sgcp = False
    cfg.use_ui_graph = False           # C5 在 Foursquare 上已证持续有害，跨域同样关闭
    cfg.sem_feat_mode = "none"         # 跨域无物品文本 → 语义特征关闭（w/o LLM-text 消融）
    cfg.use_semantic_edges = False     # 语义边关闭
    cfg.gate_mode = "context"
    # 基础通道集与 Foursquare 保持一致；cnt/rec 是否保留由 main() 按**实测重访率**自动决定
    # （见 main 中的 _auto_prior_channels）——Gowalla 重访 0.6759 属重访主导域，保留；
    # Steam/MovieLens 重访 0.0000，自动剔除。不写死在数据集分支里，规则统一可复核。
    cfg.prior_channels = "cnt,rec,pop,cooc"
    # 【必须与 Foursquare 生产配置一致】打分器/池化路径。config.py 的默认值 mlp/gru 是
    # 早期已弃用的路径（2026-07-29 消融结论：mlp/gru 显著劣于 dot/mean，后续实验一律用
    # dot+mean）。跨域若沿用默认，等于"换了个方法去做跨数据集验证"，对比无效。
    cfg.scorer = "dot"
    cfg.session_pool = "mean"
    cfg.cat_dim = 0
    cfg.use_category_edges = False
    cfg.use_geo_edges = False
    cfg.geo_radius_km = 2.0
    cfg.semantic_sim_thr = 0.30
    if name == "ml1m":
        cfg.use_category_edges = True  # MovieLens 有真实类目（genres）→ 类目边保留
        cfg.cat_dim = 16
    elif name == "gowalla":
        cfg.use_geo_edges = True       # Gowalla 有真实经纬度 → 地理边保留
        cfg.geo_radius_km = 1.0
    elif name in ("steam", "steam200k", "amazon_beauty"):
        pass                           # covisit-only 图（无 geo/类目/语义；文本仅供 LLM4POI-style 基线）
    # foursquare 不在此处理（沿用原有全特征配置）
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--dataset", default="foursquare",
                    choices=["foursquare", "ml1m", "gowalla", "steam",
                             "steam200k", "amazon_beauty"],
                    help="数据集：foursquare=原 LLM4POI-NYC；ml1m/gowalla/steam=SASRec 基准"
                         "（跨域，统一以 w/o LLM-text 模式评测：语义边/语义特征关闭，"
                         "仅保留行为先验 C6 + 共现 cooc 通道 +（可选）地理/类目边）")
    ap.add_argument("--ds_max_pois", type=int, default=0,
                    help="跨域数据集 POI 子采样上限（0=加载器默认 5000）。用于快速 pilot。")
    ap.add_argument("--ds_max_users", type=int, default=0,
                    help="跨域数据集用户子采样上限（0=加载器默认 20000）。用于快速 pilot。")
    ap.add_argument("--max_train_samples", type=int, default=0,
                    help="训练样本数上限（0=不限）。hist_mode=user 在稠密消费域会生成数十万样本"
                         "（ML-1M 达 466341），单卡跑不完 6 模型。按 cfg.seed 均匀随机下采样，"
                         "ours 与全部基线共用同一批样本；测试集不裁剪。采样比例写入结果 JSON。")
    ap.add_argument("--processed_dir", default=None)
    ap.add_argument("--out", default="head_to_head.json")
    ap.add_argument("--sota_preds", default=None,
                    help="JSON: {\"ModelName\": [[score_i for i in 0..num_pois-1], ...]}, "
                         "每行对应一个测试样本（顺序同 test_pairs.json）。")
    ap.add_argument("--max_pois", type=int, default=0, help="0=不抽样；>0 则抽高频 POI 子集")
    ap.add_argument("--no_user", action="store_true",
                    help="消融：禁用用户长期偏好模块（仅 session/KG），用于对照 C4 贡献")
    ap.add_argument("--cold_poi_ratio", type=float, default=0.0,
                    help="严格 POI 冷启动比例(0~1)：把该比例的最低频 POI 从训练剔除、只在测试做目标；"
                         "用于证明 ours(KG) 相对 CF 在冷启 POI 上的优势")
    ap.add_argument("--no_ui_graph", action="store_true",
                    help="消融：禁用 User-POI 双图高阶传播（C5），仅保留 KG + session，用于对照 C5 贡献")
    ap.add_argument("--scorer", default=None, choices=["mlp", "dot", "attn"],
                    help="打分器 mlp(concat+MLP) | dot(CF 式点积 u·v) | attn(候选-历史语义交叉注意力，C3 意图推理实例化)")
    ap.add_argument("--cooc_agg", default=None, choices=["max", "sum"],
                    help="C6-cooc 通道聚合方式。默认 None=按训练端历史均长自动选择"
                         "（<20 用 max，≥20 用 sum）；显式指定则关闭自动适配（供消融）。")
    ap.add_argument("--session_pool", default=None, choices=["gru", "mean"],
                    help="根因调试：会话编码 gru(当前) | mean(均值池化，CF 式用户表征)")
    ap.add_argument("--use_cf_score", action="store_true",
                    help="根因调试：加纯 CF 协同打分项 cf_gate*(cf_emb[cand]·session_h)")
    ap.add_argument("--cf_gate_init", type=float, default=0.1,
                    help="根因调试：cf 打分项初值权重")
    ap.add_argument("--ours_only", action="store_true",
                    help="诊断加速：只训练/评估 ours，跳过 4 基线+Pop（基线数值固定，可从其它 run 复用）")
    ap.add_argument("--use_bge", action="store_true",
                    help="激活 C1：用本地 BGE 真实语义嵌入取代 MD5 哈希占位（语义边/语义表征具真实世界知识）")
    ap.add_argument("--use_sgcp", action="store_true",
                    help="启用语义门控协同传播(SGCP)：covisit 协同信号经 C1 语义相似度门控后再传播")
    ap.add_argument("--sgcp_scale", type=float, default=None,
                    help="SGCP 门控缩放(斜率)；越大门控越像硬阈值→仅语义近邻的协同信号通过→排序更锐→NDCG 更高")
    ap.add_argument("--sgcp_bias", type=float, default=None,
                    help="SGCP 门控偏置（初始大→gate≈1 中性；训练后压低语义无关共现噪声边）")
    ap.add_argument("--max_degree", type=int, default=None,
                    help="每类关系每 POI 保留的最大邻居数（k-NN 剪枝，0=不剪枝，默认 10）")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="训练批大小。图传播每批只算一次、与批大小无关，故增大批可显著摊薄传播开销")
    ap.add_argument("--lr", type=float, default=None, help="学习率（增大批时需同步上调）")
    ap.add_argument("--homo_gnn", action="store_true",
                    help="C2 消融：四类边合并为并集图并共享同一个 W（去掉类型专属变换与类型级融合），"
                         "其余（SGCP 门控、残差、边集合）不变，用于隔离『异构传播』的净贡献")
    ap.add_argument("--no_residual", action="store_true",
                    help="C2 消融：去掉 skip(base_feat) 残差与层间恒等项，用于验证"
                         "『无残差→POI 嵌入梯度消失、表征方差塌缩（过平滑）』的论断")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子（确定性播种 + 多种子显著性验证）")
    ap.add_argument("--neg_samples", type=int, default=None,
                    help="训练时每个正样本的均匀随机负样本数（InfoNCE 负样本数 K）：K 越大→InfoNCE 越接近全 softmax→"
                         "全条件分布估计越准→分数全局校准越好→pct_rank1/NDCG 越高（默认 10）。"
                         "注意：必须保持【均匀随机】采样（勿与 --hard_neg_ratio 混用），否则偏离真实分布会破坏校准→塌缩")
    ap.add_argument("--loss", default=None, choices=["ce", "bpr", "list"],
                    help="训练目标：ce(带负采样 softmax 分类=InfoNCE) | bpr(成对排序损失，本架构已证不兼容→分数塌缩，不建议) | "
                         "list(ListNet 式 listwise：温度锐化 softmax 直接优化顶部排序/NDCG)")
    ap.add_argument("--tau", type=float, default=None,
                    help="listwise/温度锐化系数（仅 --loss list 生效）：sc/tau 后再算 softmax；"
                         "tau<1 越锐→顶部排序越自信→NDCG 越高（默认 0.5；CE 等价 tau=1）")
    ap.add_argument("--hard_neg_ratio", type=float, default=None,
                    help="C1 语义 hard-negative mining：训练候选中语义近邻负样本占比(0~1)；"
                         ">0 时用语义相似 POI 替换该比例的随机负样本，逼模型分离近似重复 POI→抬 NDCG；"
                         "保留 CE 损失以维持全局校准（默认 0=纯随机负样本）")
    ap.add_argument("--hard_neg_topk", type=int, default=None,
                    help="每个 POI 取语义 top-k 近邻作为 hard 负样本候选池（仅 --hard_neg_ratio>0 生效，默认 50）")
    ap.add_argument("--bge_model_dir", default="bge_model",
                    help="本地 bge 模型目录（sentence-transformers 格式）")
    ap.add_argument("--bge_cache", default="poi_bge_emb.npy",
                    help="离线分块编码的 POI BGE 嵌入缓存（避免训练时一次性编码触发 segfault）")
    ap.add_argument("--sem_feat_mode", default=None, choices=["bge", "none", "cat_onehot"],
                    help="C1 贡献拆分（节点特征路径）：bge(默认,真实BGE向量) | none(节点特征不含语义向量，"
                         "仅保留语义边+SGCP) | cat_onehot(用类目 one-hot 替代 BGE，"
                         "『类目名泄漏』对照——本数据集 POI 文本为 '{cat} near {lat},{lng}' 合成，"
                         "若该模式与 bge 相当，说明 C1 增益主要来自类目先验)。"
                         "三种模式下语义边与 SGCP 门控均使用原始 BGE 向量，保证单变量对照")
    ap.add_argument("--mask_hist", default="auto", choices=["auto", "on", "off"],
                    help="【评测协议】是否把历史已交互物品从全候选中屏蔽。"
                         "auto(默认)=按测试端 revisit_ratio 自动判定(<0.05 则开)：ml1m/steam/gowalla "
                         "等无重复消费域开启（SASRec 一系工作的标准协议），Foursquare-NYC 这类重访主导域 "
                         "(revisit=0.7574) 关闭，否则会屏蔽掉正确答案本身。on/off=手动强制。"
                         "该开关对 ours 与全部基线统一施加，且必须无条件屏蔽（不能『除目标外』，那是标签泄漏）")
    ap.add_argument("--sem_thr", type=float, default=None,
                    help="语义边余弦阈值（覆盖 config.semantic_sim_thr）。BGE 真实嵌入相似度普遍偏高，"
                         "需显著高于 MD5 占位的 0.30，例如 0.85~0.92 才能得到稀疏有意义的语义边")
    ap.add_argument("--prior_channels", default=None,
                    help="C6 行为先验通道，逗号分隔子集(cnt/rec/pop)，例如 'cnt,rec,pop'。"
                         "cnt=历史出现次数(含 History-Frequency 规则) rec=近因(含 History-Recency) "
                         "pop=全局热度(含 Popularity)。空/不传=停用，保持旧行为。"
                         "本基准 75.7% 目标为重访，缺此通道的打分头无法表达『该用户来过这里 N 次』")
    ap.add_argument("--gate_mode", default=None, choices=["context", "global", "off"],
                    help="C6 门控：context=由会话表征+重复度统计逐样本产生通道权重(默认推荐) | "
                         "global=全局可学习标量权重(消融，测上下文依赖的净贡献) | off=停用先验通道")
    ap.add_argument("--hist_mode", default=None, choices=["trajectory", "user"],
                    help="训练样本的历史构造方式。trajectory=会话内前缀（旧默认，训练历史均值 7.8、"
                         "重访率 38.6%%）；user=同一用户各会话按时间拼接后滑窗（与官方测试协议一致，"
                         "测试端历史均值 143.2、重访率 75.7%%）。二者的训练/测试分布错配是本工作发现的"
                         "关键实验缺陷，建议配合 --seq_len 一起调大")
    ap.add_argument("--seq_len", type=int, default=None,
                    help="历史窗口上限（覆盖 config.seq_len，默认 20）。hist_mode=user 时建议 200")
    ap.add_argument("--no_graph", action="store_true",
                    help="C2 净贡献对照：完全跳过异构消息传递，只保留 skip 线性投影。"
                         "节点特征编码器/隐层维度/打分头/训练目标不变，唯一变量是有无图传播")
    ap.add_argument("--no_geo_edges", dest="no_geo_edges", action="store_true",
                    help="P1-5 单边缘类型消融：跳过地理邻近边。")
    ap.add_argument("--no_category_edges", dest="no_category_edges", action="store_true",
                    help="P1-5 单边缘类型消融：跳过类目边。")
    ap.add_argument("--no_semantic_edges", dest="no_semantic_edges", action="store_true",
                    help="P1-5 单边缘类型消融：跳过语义边（需 --use_bge 激活真实 BGE 嵌入时才有意义）。")
    ap.add_argument("--no_covisit_edges", dest="no_covisit_edges", action="store_true",
                    help="P1-5 单边缘类型消融：跳过纯行为共访边，构造 geo/cat/sem-only 图。")
    ap.add_argument("--num_gnn_layers", type=int, default=None,
                    help="P1-5 传播深度消融：异构 GNN 传播层数 L（覆盖 config.num_gnn_layers，默认 2）。"
                         "L=1=仅一层类型专属消息传递；L=2=残差叠加第二层。")
    ap.add_argument("--covisit_score", default=None, choices=["raw", "cosine", "pmi"],
                    help="共访边选边打分：raw=原始共现次数(旧行为) | cosine | pmi(去热度偏置)。"
                         "稠密消费域(Steam/ML-1M)必须去偏，否则前 10 个热门物品独占 6~8 成的边、"
                         "任意两节点 top-10 邻居重合 32~55%%，均值聚合后表征塌缩(余弦 0.9997)。"
                         "不指定时按训练端图统计自动选择")
    ap.add_argument("--repr_center", action="store_true",
                    help="对 POI 最终表征做跨节点去均值。诊断依据：Steam 上 poi_repr_var_mean=8.85 "
                         "但 pairwise_cos=0.9997，即公共均值 μ 淹没差异 δ，点积退化为与用户无关的"
                         "物品偏置（KG 通道门控被压到 0.0345）。LayerNorm 不去除跨节点公共分量，"
                         "本开关补上这一刀，使打分严格等于 δ_u·δ_c")
    ap.add_argument("--no_kg_channel", action="store_true",
                    help="C6 致命消融：把整条 KG 语义打分通道从融合中移除(不参与 stack，而非权重置零)，"
                         "只留行为先验通道。若该消融与完整模型指标持平，说明本基准上 LLM/KG 净增量为零，"
                         "必须在论文中如实报告")
    ap.add_argument("--save_model", default=None,
                    help="训练后保存 ours 模型权重到该路径(.pt)；用于后续 --eval_only 复用已训练模型做诊断，免重训")
    ap.add_argument("--eval_only", action="store_true",
                    help="跳过训练，直接加载 --load_model 指定的 ours 权重做评估（配合 --save_model 产物）")
    ap.add_argument("--load_model", default=None,
                    help="--eval_only 时加载的 ours 权重路径（须与 --scorer/--use_sgcp/--use_bge 等结构一致）")
    args = ap.parse_args()

    # ---------- 确定性播种（多种子显著性验证 + 可复现）----------
    cfg_seed = args.seed if getattr(args, "seed", None) is not None else 42
    torch.manual_seed(cfg_seed)
    np.random.seed(cfg_seed)
    random.seed(cfg_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = Config()
    cfg.seed = cfg_seed
    cfg.device = args.device
    cfg.epochs = args.epochs
    if args.num_gnn_layers is not None:
        cfg.num_gnn_layers = args.num_gnn_layers
    if args.no_ui_graph:
        cfg.use_ui_graph = False
    if args.homo_gnn:
        cfg.homo_gnn = True
    if args.no_residual:
        cfg.use_residual = False
    if args.max_degree is not None:
        cfg.max_degree = args.max_degree
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.scorer:
        cfg.scorer = args.scorer
    if args.session_pool:
        cfg.session_pool = args.session_pool
    if args.use_cf_score:
        cfg.use_cf_score = True
        cfg.cf_gate_init = args.cf_gate_init
    if args.use_bge:
        cfg.use_bge = True
        cfg.bge_model_dir = args.bge_model_dir
        cfg.bge_cache = args.bge_cache
        cfg.sem_dim = 768   # bge-base 输出维度（动态 feat_dim 已兼容）
    if args.sem_thr is not None:
        cfg.semantic_sim_thr = args.sem_thr
    if args.sem_feat_mode is not None:
        cfg.sem_feat_mode = args.sem_feat_mode
    if args.prior_channels is not None:
        cfg.prior_channels = args.prior_channels
    if args.gate_mode is not None:
        cfg.gate_mode = args.gate_mode
    if args.no_kg_channel:
        cfg.use_kg_channel = False
    if args.no_graph:
        cfg.no_graph = True
    if args.repr_center:
        cfg.repr_center = True
    if args.hist_mode is not None:
        cfg.hist_mode = args.hist_mode
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if args.use_sgcp:
        cfg.use_sgcp = True
    if args.sgcp_scale is not None:
        cfg.sgcp_scale = args.sgcp_scale
    if args.sgcp_bias is not None:
        cfg.sgcp_bias = args.sgcp_bias
    if args.loss:
        cfg.loss_type = args.loss
    if args.tau is not None:
        cfg.list_tau = args.tau
    if args.hard_neg_ratio is not None:
        cfg.hard_neg_ratio = args.hard_neg_ratio
    if args.hard_neg_topk is not None:
        cfg.hard_neg_topk = args.hard_neg_topk
    if args.neg_samples is not None:
        cfg.neg_samples = args.neg_samples
    # P1-5 单边缘类型消融：geo/cat/sem/covisit 关闭开关（no_*=True 时显式关掉对应边，
    # 默认 False = 保留 config 默认 True）。用独立 no_* dest 避免 argparse 共享 dest 的 default 竞态。
    if getattr(args, "no_geo_edges", False):
        cfg.use_geo_edges = False
    if getattr(args, "no_category_edges", False):
        cfg.use_category_edges = False
    if getattr(args, "no_semantic_edges", False):
        cfg.use_semantic_edges = False
    if getattr(args, "no_covisit_edges", False):
        cfg.use_covisit_edges = False

    # 1. 加载数据（按 --dataset 路由）
    cooc_matrix = None
    if args.dataset == "foursquare":
        pois, checkins, test_samples, num_pois, stats, cold_pois = load_real_nyc(
            args.processed_dir, args.cold_poi_ratio)
        print(f"[data] Foursquare-NYC：POI={num_pois} 训练轨迹={len(checkins)} "
              f"测试样本={len(test_samples)} | stats={stats}")
    else:
        apply_dataset_cfg(cfg, args.dataset)
        # 【顺序修复】apply_dataset_cfg 在 CLI 覆盖之后执行，会把用户显式传入的开关冲掉，
        # 导致 --prior_channels / --gate_mode / --sem_feat_mode 等消融参数静默失效
        # （表现为"传了参数但结果与默认完全一致"，极难察觉）。此处把显式 CLI 参数重新压上，
        # 保证优先级 = CLI > 数据集默认 > config 默认。
        for _a, _k in (("prior_channels", "prior_channels"), ("gate_mode", "gate_mode"),
                       ("sem_feat_mode", "sem_feat_mode"), ("hist_mode", "hist_mode"),
                       ("scorer", "scorer"), ("session_pool", "session_pool"),
                       ("cooc_agg", "cooc_agg"), ("covisit_score", "covisit_score"),
                       ("sem_thr", "semantic_sim_thr")):
            _v = getattr(args, _a, None)
            if _v is not None:
                setattr(cfg, _k, _v)
                print(f"[cfg] CLI 覆盖数据集默认: {_k}={_v}")
        args.max_pois = 0  # 子采样已由通用加载器完成，跳过 head_to_head 二次抽样
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
        mp = args.ds_max_pois if args.ds_max_pois > 0 else GL.DEFAULT_MAX_POIS
        mu = args.ds_max_users if args.ds_max_users > 0 else GL.DEFAULT_MAX_USERS
        if args.dataset == "ml1m":
            pois, checkins, test_samples, num_pois, stats, cold_pois, cooc_matrix = \
                GL.load_movielens(os.path.join(root, "ml-1m", "ml-1m"),
                                  max_pois=mp, max_users=mu,
                                  out_dir=os.path.join(root, "ml-1m", "processed"), name="movielens-1m")
        elif args.dataset == "gowalla":
            gz = os.path.join(root, "gowalla", "loc-gowalla_totalCheckins.txt.gz")
            pois, checkins, test_samples, num_pois, stats, cold_pois, cooc_matrix = \
                GL.load_gowalla(gz, max_pois=mp, max_users=mu,
                                out_dir=os.path.join(root, "gowalla", "processed"), name="gowalla")
        elif args.dataset == "steam":  # 600-POI SASRec-Steam（无文本，no-text 对照）
            st = os.path.join(root, "steam", "Steam.txt")
            pois, checkins, test_samples, num_pois, stats, cold_pois, cooc_matrix = \
                GL.load_steam(st, max_pois=mp, max_users=mu,
                              out_dir=os.path.join(root, "steam", "processed"), name="steam")
        elif args.dataset == "steam200k":
            s2 = os.path.join(root, "steam200k", "steam-200k.csv")
            pois, checkins, test_samples, num_pois, stats, cold_pois, cooc_matrix = \
                GL.load_steam200k(s2, max_pois=mp, max_users=mu,
                                  out_dir=os.path.join(root, "steam200k", "processed"), name="steam200k")
        elif args.dataset == "amazon_beauty":
            meta = os.path.join(root, "amazon_beauty", "meta_All_Beauty.parquet")
            csvs = [os.path.join(root, "amazon_beauty", f"All_Beauty.{s}.csv")
                    for s in ("train", "valid", "test")]
            pois, checkins, test_samples, num_pois, stats, cold_pois, cooc_matrix = \
                GL.load_amazon_beauty(meta, csvs, max_pois=mp, max_users=mu, min_freq=5,
                                      out_dir=os.path.join(root, "amazon_beauty", "processed"),
                                      name="amazon_beauty")
        print(f"[data] {args.dataset}（跨域 w/o LLM-text）：POI={num_pois} 训练轨迹={len(checkins)} "
              f"测试样本={len(test_samples)} | cooc={'%dx%d' % cooc_matrix.shape if cooc_matrix is not None else None} "
              f"| stats={stats}")

    # 严格冷启动实验需全量 POI 候选，强制关闭高频抽样
    if args.cold_poi_ratio and args.cold_poi_ratio > 0 and args.max_pois > 0:
        print(f"[cold-POI] 强制 max_pois=0（冷启动实验需全量 POI 候选）")
        args.max_pois = 0

    # 可选：抽高频 POI 子集（控制图规模）
    if args.max_pois and args.max_pois < num_pois:
        from collections import Counter
        cnt = Counter(p for _, seq in checkins for p in seq)
        keep = set(i for i, _ in cnt.most_common(args.max_pois))
        new_id = {p: k for k, p in enumerate(keep)}
        pois = [pois[p] for p in keep]
        for i, m in enumerate(pois):
            m["poi_id"] = i
        checkins = [(u, [new_id[p] for p in seq if p in new_id]) for u, seq in checkins]
        checkins = [(u, s) for u, s in checkins if len(s) >= 2]
        test_samples = [(u, [new_id[p] for p in h if p in new_id], new_id[t])
                        for u, h, t in test_samples if t in new_id and all(p in new_id for p in h)]
        num_pois = args.max_pois
        print(f"[data] 抽样到 max_pois={num_pois}")

    # 2. 训练样本（全量训练集，不做验证划分）
    users = list({u for u, _ in checkins})
    n_users = max(users) + 1
    train_samples = _build_samples(checkins, cfg.seq_len, set(users),
                                   hist_mode=getattr(cfg, "hist_mode", "trajectory"))
    # 2b.【训练样本预算】hist_mode=user 会为每个交互生成一个样本，稠密消费域会爆炸：
    #     MovieLens-1M（800 物品 / 2000 用户 / 人均历史 118.5）→ 466341 个训练样本，
    #     ×6 个模型 ×6 轮在单卡上不可行。此处按固定 seed 均匀随机下采样到预算内，
    #     对 ours 与全部基线使用【同一批】样本，保证跨模型可比；测试集不做任何裁剪。
    #     采样比例写入 train_diag（train_sample_ratio），论文须如实披露。
    _n_full = len(train_samples)
    _budget = int(getattr(args, "max_train_samples", 0) or 0)
    if _budget > 0 and _n_full > _budget:
        _rng = random.Random(cfg.seed)
        train_samples = _rng.sample(train_samples, _budget)
        print(f"[samples] 训练样本下采样: {_n_full} -> {len(train_samples)} "
              f"(budget={_budget}, seed={cfg.seed}, ratio={len(train_samples)/_n_full:.4f})")
    _train_sample_ratio = round(len(train_samples) / max(_n_full, 1), 4)
    _hl = [len(h) for _, h, _ in train_samples]
    _rv = sum(1 for _, h, t in train_samples if t in set(h))
    _revisit_ratio_train = round(_rv / max(len(train_samples), 1), 4)
    # 测试端重访率：目标是否已出现在该样本历史中。Foursquare-NYC=0.7574（重访主导），
    # 而 MovieLens/Steam 等无重复消费域≈0 —— 这是判定"复读历史类平凡基线是否可行"的关键统计量，
    # 必须随结果一并报告，否则跨数据集比较会被重访红利污染。
    _rv_te = sum(1 for _, h, t in test_samples if t in set(h))
    _revisit_ratio_test = round(_rv_te / max(len(test_samples), 1), 4)
    _hist_len_mean_test = round(sum(len(h) for _, h, _ in test_samples) / max(len(test_samples), 1), 1)
    print(f"[samples] hist_mode={getattr(cfg, 'hist_mode', 'trajectory')} "
          f"seq_len={cfg.seq_len} n={len(train_samples)} "
          f"hist_len_mean={sum(_hl)/max(len(_hl),1):.1f} "
          f"revisit_ratio={_revisit_ratio_train:.4f} "
          f"| 测试端: hist_len_mean={_hist_len_mean_test} revisit_ratio={_revisit_ratio_test:.4f}")

    # 2c. 【评测协议】历史屏蔽开关——由测试端重访率自动决定，对 ours 与全部基线一视同仁。
    #     无重复消费域（ml1m/steam/gowalla，revisit≈0）：屏蔽历史已交互物品，这是 SASRec
    #     一系工作的标准协议；不屏蔽会让几十个历史物品占满 top-10，Recall 被系统性压低
    #     （实测 Steam 上 Popularity R@10 从 0.0448 → 0.0725，1.6 倍差距）。
    #     重访主导域（Foursquare-NYC，revisit=0.7574）：绝不能开，否则会屏蔽掉正确答案本身。
    if args.mask_hist == "auto":
        _mask_hist = _revisit_ratio_test < 0.05
    else:
        _mask_hist = (args.mask_hist == "on")
    print(f"[protocol] mask_history={'ON' if _mask_hist else 'OFF'} "
          f"(依据 revisit_ratio_test={_revisit_ratio_test:.4f}, 模式={args.mask_hist}) "
          f"— 对 ours 与全部基线统一施加")
    if _mask_hist and _revisit_ratio_test >= 0.05:
        print(f"  ⚠️ 警告：重访率 {_revisit_ratio_test:.4f} 不可忽略却强开屏蔽，"
              f"约 {_revisit_ratio_test*100:.1f}% 的样本正确答案会被屏蔽掉，指标将失真。")

    # 2d. 【C6 通道域适配】cnt/rec 通道是否保留——同样由实测重访率决定，与 mask_history 同源。
    #     判据：门控 w=softplus(·)≥0 只能非负；在 revisit≈0 的域，f_cnt/f_rec 在**所有正样本**
    #     上恒为 0，通道不含判别信息，只会给"必然不是答案"的历史物品加分，而非负门控无法
    #     学出负权重去反转 → 训练 loss 死锁（Steam 实测锁死在 2.2100，随机基准 ln(11)=2.3979）。
    #     用户显式传 --prior_channels 时不做自动调整（消融实验需要手动控制）。
    if args.dataset != "foursquare" and args.prior_channels is None:
        _ch = [c for c in str(cfg.prior_channels).split(",") if c]
        if _revisit_ratio_train < 0.05 and ("cnt" in _ch or "rec" in _ch):
            _kept = [c for c in _ch if c not in ("cnt", "rec")]
            print(f"[C6] 训练端重访率 {_revisit_ratio_train:.4f} < 0.05 → 自动剔除 cnt/rec 通道："
                  f"{','.join(_ch)} → {','.join(_kept)}（正样本上二者恒为 0，非负门控无法利用）")
            cfg.prior_channels = ",".join(_kept)
        else:
            print(f"[C6] 训练端重访率 {_revisit_ratio_train:.4f} ≥ 0.05 → 保留全部通道 "
                  f"{','.join(_ch)}（重访主导域，cnt/rec 含判别信息）")

    # 2e. 【C6-cooc 聚合方式域适配】由实测**历史均长**决定 max / sum，与上面两项同为"域适配"。
    #     判据：cooc 特征 = 候选的 top-k 共现邻居与本样本历史求交后聚合。
    #       · 短会话（Foursquare trajectory 均长 7.8）：命中数通常 0~2，max 与 sum 近似等价，
    #         而 max 对偶发噪声共现更稳健 → 沿用 max（保持既有全部 Foursquare 结果不变）。
    #       · 长稠密历史（Steam 74.5 / Gowalla 58.4）：几乎每个候选都能命中若干强共现邻居，
    #         max 饱和到 ≈1.0，候选大面积并列而丧失区分度。实测后果：Steam 上 ours 精确退化
    #         到热度水平（R@10=0.0729 vs Pop 0.0725），够不着同为共现规则的 ItemKNN（0.0809）。
    #         → 改用 log1p(sum)，与 ItemKNN 的 Σ 同构，模型才可能在其之上再叠加 KG/序列信号。
    #     阈值 20 取在两类分布中间（7.8 vs 58.4，无歧义），显式传 --cooc_agg 时不干预。
    _hist_len_mean_train = sum(_hl) / max(len(_hl), 1)
    if args.cooc_agg is None and "cooc" in str(cfg.prior_channels):
        if _hist_len_mean_train >= 20.0 and str(getattr(cfg, "cooc_agg", "max")) != "sum":
            cfg.cooc_agg = "sum"
            print(f"[C6] 训练端历史均长 {_hist_len_mean_train:.1f} ≥ 20 → cooc 聚合 max → sum"
                  f"（长历史下 max 饱和，候选并列丧失区分度）")
        else:
            print(f"[C6] 训练端历史均长 {_hist_len_mean_train:.1f} → cooc 聚合沿用 "
                  f"{getattr(cfg, 'cooc_agg', 'max')}")

    # 2f. 【C2 选边去热度偏置】由实测**共访图密度**决定 raw / pmi，是 C2 侧的域适配。
    #     判据：kg_builder 按 covisit_score 给每个 POI 挑 top-k 共访邻居。用原始共现次数
    #     (raw) 排序时，任意物品与头部热门物品的共现次数都很大 → 所有节点挑到同一批枢纽
    #     → 均值聚合后消息几乎相同 → 表征塌缩、梯度饥饿、KG 通道被门控关死。
    #     hub_collapse_probe.py 实测（每节点 top-10）：
    #       Steam(密度1.000)  raw: 邻居Jaccard 0.5527 / 枢纽覆盖 77.6% / 邻居多样性 11.5%
    #                         pmi: 0.0295 / 14.7% / 80.7%
    #       ML-1M(密度1.000)  raw: 0.3160 / 60.7% / 16.5%   pmi: 0.0179 / 6.8% / 74.1%
    #       Gowalla(密度0.244) raw: 0.0396 / 18.5% / 71.0%  ← 本就健康，无需去偏
    #     端到端佐证：Steam 上 --no_graph 反而更好（R@10 0.0865 vs 0.0809），余弦
    #     0.9997→0.5618、梯度 8.75e-05→1.34e-02，说明该域的图传播在 raw 选边下是净负贡献。
    #     阈值 0.5 取在 0.244 与 1.000 之间（无歧义）；显式传 --covisit_score 时不干预。
    if args.covisit_score is None and cooc_matrix is not None:
        try:
            import numpy as _np
            _cm = _np.asarray(cooc_matrix)
            _n = int(_cm.shape[0])
            _dens = float((_cm > 0).sum() - _np.count_nonzero(_np.diag(_cm))) / \
                max(_n * (_n - 1), 1)
        except Exception:
            _dens = 0.0
        if _dens > 0.5:
            cfg.covisit_score = "pmi"
            print(f"[C2] 共访图密度 {_dens:.3f} > 0.5 → 选边打分 raw → pmi"
                  f"（稠密消费域，raw 会让前 10 个热门物品垄断 6~8 成的边 → 表征塌缩）")
        else:
            print(f"[C2] 共访图密度 {_dens:.3f} ≤ 0.5 → 选边打分沿用 "
                  f"{getattr(cfg, 'covisit_score', 'raw')}（稀疏域，邻居本就多样）")

    # 3. ours
    # 构建 User-POI 二部图边（C5 双图高阶传播用）；若关闭 C5 则不构建
    ui_edge = None
    if getattr(cfg, "use_ui_graph", True) and n_users > 0:
        ui_edge = build_ui_edge(checkins, num_pois)
        print(f"[C5] 构建 User-POI 边: {ui_edge.shape[1]} 条")
    from .model.stkg_net import STKGNet
    from .train import eval_ours_full
    if args.eval_only and args.load_model:
        # 跳过训练：直接加载已保存权重做评估（诊断复用，免重训）
        # 【修复】此前此处按 STKGNet(cfg, num_pois, len(pois), n_users=...) 构造，
        # 位置参数与签名 (cfg, num_pois, num_cats, cat_ids, sem_vecs, edge_index, ...) 不匹配，
        # 必然 TypeError；且未重建 KG。现改为先按同参数重建 KG 再构造，结构与训练完全一致。
        print(f"\n=== [eval_only] 加载 ours 权重: {args.load_model} ===")
        kg = build_kg(cfg, pois, checkins)
        print("[KG] 边统计:", kg.stats())
        ours_model = STKGNet(cfg, num_pois, kg.num_cats, kg.cat_ids, kg.sem_vecs, kg.edge_index,
                             n_users=(0 if args.no_user else n_users),
                             user_item_edge=ui_edge,
                             pop_prior=build_pop_prior(checkins, num_pois),
                             cooc_matrix=cooc_matrix).to(args.device)
        ours_model.load_state_dict(torch.load(args.load_model, map_location=args.device))
        ours_model.eval()
    else:
        print("\n=== 训练 LLM-STKG (ours) ===")
        ours_model = train_ours(cfg, pois, checkins, train_samples, num_pois, args.device,
                                n_users=(0 if args.no_user else n_users),
                                user_item_edge=ui_edge, cooc_matrix=cooc_matrix)
        if args.save_model:
            torch.save(ours_model.state_dict(), args.save_model)
            print(f"[save_model] 已保存 ours 权重 -> {args.save_model}")
    ours_metrics = eval_ours_full(ours_model, test_samples, num_pois, cfg, args.device,
                                  mask_hist=_mask_hist)
    results = {"LLM-STKG (ours)": ours_metrics}

    # 4. 基线（session-based 公平评估）
    from collections import Counter
    train_freq = Counter(p for _, seq in checkins for p in seq)
    pop_score = torch.zeros(num_pois, dtype=torch.float32)
    for p, c in train_freq.items():
        if p < num_pois:
            pop_score[p] = float(c)
    if not args.ours_only:
        print("\n=== 训练并评估基线（session-based）===")
        # 注：max_pois 抽样可能过滤掉部分用户（序列变短），users 集合变小，
        # 但保留下来的 checkins 仍含原重映射 uid（最大值可能 > len(users)），
        # 故用 max+1 建表，避免 user embedding 索引越界。
        baselines = build_baselines(n_users, num_pois, device=args.device)
        for name, m in baselines.items():
            print(f"--- {name} ---")
            m.fit(train_samples, epochs=cfg.epochs, device=args.device)
            results[name] = eval_session(m, test_samples, num_pois, args.device,
                                         mask_hist=_mask_hist)

        # 4b. 热度基线（Popularity）：仅按训练集 POI 频次排名，完全不依赖历史序列。
        #     作用：检验 CF 类基线（LightGCN/BPR-MF）的高分是否只是"记忆热门 POI"。
        #     若 Pop 也接近 0.4，则其 R@10≈0.47 几乎不含模型贡献，不可作为 ours 的有效对照。
        scores_pop = pop_score.unsqueeze(0).repeat(len(test_samples), 1)
        tgts_all = torch.tensor([t for _, _, t in test_samples], dtype=torch.long)
        if _mask_hist:
            scores_pop = mask_history(scores_pop, test_samples, num_pois)
        results["Popularity (Pop)"] = rank_metrics(scores_pop, tgts_all, k_list=(5, 10))
        print(f"[Popularity] R@10={results['Popularity (Pop)']['Recall@10']:.4f} "
              f"(若≈CF基线，则 CF 高分源于热度偏差而非模型能力)")

    # 4c. 冷启动子集（测试目标在训练中低频 ≤5 次）——ours 靠 LLM 语义+KG 应占优
    cold_samples = [s for s in test_samples if train_freq.get(s[2], 0) <= 5]
    print(f"\n[冷启动] 测试样本中目标低频(≤5)数量: {len(cold_samples)}/{len(test_samples)}")
    results_cold = {}
    if cold_samples:
        results_cold["LLM-STKG (ours)"] = eval_ours_full(
            ours_model, cold_samples, num_pois, cfg, args.device, mask_hist=_mask_hist)
        if not args.ours_only:
            for name, m in baselines.items():
                results_cold[name] = eval_session(m, cold_samples, num_pois, args.device,
                                                  mask_hist=_mask_hist)
            scores_pop_cold = pop_score.unsqueeze(0).repeat(len(cold_samples), 1)
            tgts_cold = torch.tensor([t for _, _, t in cold_samples], dtype=torch.long)
            if _mask_hist:
                scores_pop_cold = mask_history(scores_pop_cold, cold_samples, num_pois)
            results_cold["Popularity (Pop)"] = rank_metrics(scores_pop_cold, tgts_cold, k_list=(5, 10))

    # 4d. 严格 POI 冷启动子集（目标 POI 训练阶段完全不可见）——ours 靠 KG 类别/地理/语义应占优，CF 哑火
    cold_poi_samples = [s for s in test_samples if s[2] in cold_pois]
    print(f"\n[严格冷启动 POI] 测试样本中目标为冷启 POI 数量: {len(cold_poi_samples)}/{len(test_samples)}")
    results_coldpoi = {}
    if cold_poi_samples:
        results_coldpoi["LLM-STKG (ours)"] = eval_ours_full(
            ours_model, cold_poi_samples, num_pois, cfg, args.device, mask_hist=_mask_hist)
        if not args.ours_only:
            for name, m in baselines.items():
                results_coldpoi[name] = eval_session(m, cold_poi_samples, num_pois, args.device,
                                                     mask_hist=_mask_hist)
            scores_pop_cp = pop_score.unsqueeze(0).repeat(len(cold_poi_samples), 1)
            tgts_cp = torch.tensor([t for _, _, t in cold_poi_samples], dtype=torch.long)
            if _mask_hist:
                scores_pop_cp = mask_history(scores_pop_cp, cold_poi_samples, num_pois)
            results_coldpoi["Popularity (Pop)"] = rank_metrics(scores_pop_cp, tgts_cp, k_list=(5, 10))

    # 5. SOTA 预测（若有）
    if args.sota_preds:
        print(f"\n=== 接入 SOTA 预测：{args.sota_preds} ===")
        with open(args.sota_preds, encoding="utf-8") as f:
            sota = json.load(f)
        for name, mat in sota.items():
            scores = torch.tensor(mat, dtype=torch.float32)
            tgts = torch.tensor([t for _, _, t in test_samples], dtype=torch.long)
            if _mask_hist:
                scores = mask_history(scores, test_samples, num_pois)
            m = rank_metrics(scores, tgts, k_list=(5, 10))
            m["__rank_diag__"] = rank_diag(scores, tgts)
            results[name] = m

    # 6. 输出
    # 抽取各模型的排名诊断（__rank_diag__）到独立段，避免污染打印用的 metrics 字典
    def _extract_diag(d):
        out = {}
        for k, v in list(d.items()):
            if isinstance(v, dict) and "__rank_diag__" in v:
                out[k] = v.pop("__rank_diag__")
        return out
    rank_diag_section = {"full": _extract_diag(results)}
    if cold_samples:
        rank_diag_section["cold"] = _extract_diag(results_cold)
    if cold_poi_samples:
        rank_diag_section["coldpoi"] = _extract_diag(results_coldpoi)
    payload = {
        "dataset": ({"foursquare": "Foursquare-NYC (LLM4POI, real)",
                     "ml1m": "MovieLens-1M (SASRec benchmark)",
                     "gowalla": "Gowalla (SNAP check-ins)",
                     "steam": "Steam (SASRec benchmark, no text)",
                     "steam200k": "Steam-200k (Kaggle, game-title text)",
                     "amazon_beauty": "Amazon Beauty (All_Beauty 2023, rich text)"}
                    .get(args.dataset, args.dataset)),
        "dataset_key": args.dataset,
        # 跨域数据集统一以 w/o LLM-text 模式评测（语义特征/语义边关闭），须在论文中显式标注
        "text_mode": ("LLM-text (BGE semantic)" if args.dataset == "foursquare"
                      else "w/o LLM-text (behavior + structure only)"),
        # POI 文本是否可用（供 LLM4POI-style 基线做语义种子）：ours 跨域统一走 w/o LLM-text，
        # 但需如实标注哪些域「文本可得」以解释 LLM4POI-style 的 0 / 非零差异。
        "poi_text_available": bool(args.dataset in
                                   ("foursquare", "ml1m", "steam200k", "amazon_beauty")),
        # 评测协议：历史屏蔽是否开启。跨数据集比较时必须随表报告，否则读者无法判断可比性
        "mask_history": bool(_mask_hist),
        "mask_history_mode": args.mask_hist,
        "revisit_ratio_train": _revisit_ratio_train,
        "revisit_ratio_test": _revisit_ratio_test,
        # 训练样本预算：<1.0 表示做了均匀随机下采样（ours 与全部基线共用同一批），须在论文披露
        "train_samples_full": _n_full,
        "train_samples_used": len(train_samples),
        "train_sample_ratio": _train_sample_ratio,
        "hist_len_mean_test": _hist_len_mean_test,
        "protocol": "full-candidate ranking, official train/test split, session-based baselines",
        "stats": stats,
        "num_pois": num_pois,
        "rank_diag": rank_diag_section,
        "train_diag": getattr(ours_model, "_diag", {}),
        "num_test": len(test_samples),
        "results": results,
        "cold_start(≤5)": {
            "n": len(cold_samples),
            "results": results_cold,
        },
        "cold_poi_strict": {
            "n": len(cold_poi_samples),
            "ratio": args.cold_poi_ratio,
            "results": results_coldpoi,
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("\n=== 头对头对比表（真实 Foursquare-NYC，全量）===")
    header = f"{'Model':<22}{'R@5':>9}{'R@10':>9}{'N@5':>9}{'N@10':>9}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<22}{m['Recall@5']:>9.4f}{m['Recall@10']:>9.4f}"
              f"{m['NDCG@5']:>9.4f}{m['NDCG@10']:>9.4f}")
    if cold_samples:
        print(f"\n=== 冷启动子集（目标训练频次≤5，n={len(cold_samples)}）===")
        print(header)
        print("-" * len(header))
        for name, m in results_cold.items():
            print(f"{name:<22}{m['Recall@5']:>9.4f}{m['Recall@10']:>9.4f}"
                  f"{m['NDCG@5']:>9.4f}{m['NDCG@10']:>9.4f}")
    if cold_poi_samples:
        print(f"\n=== 严格 POI 冷启动子集（目标训练完全不可见，n={len(cold_poi_samples)}）===")
        print(header)
        print("-" * len(header))
        for name, m in results_coldpoi.items():
            print(f"{name:<22}{m['Recall@5']:>9.4f}{m['Recall@10']:>9.4f}"
                  f"{m['NDCG@5']:>9.4f}{m['NDCG@10']:>9.4f}")
    print(f"\n输出已保存: {args.out}")


if __name__ == "__main__":
    main()
