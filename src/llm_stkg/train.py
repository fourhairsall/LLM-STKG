"""训练流水线：构建 KG → 拆分 → 训练（负采样 CE）→ 全候选评估。

用法：
  from llm_stkg.train import train_model
  model, metrics = train_model(cfg, pois, checkins)
"""
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .kg.kg_builder import TourismKG
from .kg.llm_interface import LLMInterface
from .model.stkg_net import STKGNet
from .evaluate import rank_metrics, rank_diag, mask_history


def _build_samples(checkins, seq_len, train_users, hist_mode="trajectory"):
    """构造 (uid, history, target) 训练样本。

    hist_mode
    ---------
    "trajectory"（旧默认）: 历史 = **当前会话内**的前缀，最长 seq_len。
        问题：本数据集每条会话平均仅 7.6 次签到，于是训练历史均值 7.8、
        目标已在历史中的比例仅 38.6%；而官方测试样本给的是用户**跨会话的全部历史**
        （均值 143.2、重访比例 75.7%）。模型在"短历史/低重访"上训练、在
        "长历史/高重访"上测试，这是我们自己引入的协议错配，会系统性低估任何
        依赖历史统计的能力。
    "user": 把同一用户的各会话按时间顺序拼接成完整签到序列后再滑窗，历史最长 seq_len。
        与官方测试协议一致。加载器保证同一用户的会话在 checkins 中已按时间升序
        （已验证 1047 个用户全部有序），故直接顺序拼接即可。
    """
    samples = []
    if hist_mode == "user":
        order, seen, per_user = [], set(), {}
        for uid, seq in checkins:
            if uid not in train_users:
                continue
            if uid not in seen:
                seen.add(uid)
                order.append(uid)
                per_user[uid] = []
            per_user[uid].extend(seq)
        for uid in order:
            full = per_user[uid]
            for t in range(1, len(full)):
                samples.append((uid, full[max(0, t - seq_len):t], full[t]))
        return samples
    for uid, seq in checkins:
        if uid not in train_users:
            continue
        for t in range(1, len(seq)):
            hist = seq[max(0, t - seq_len):t]
            samples.append((uid, hist, seq[t]))
    return samples


class TrajDataset(Dataset):
    def __init__(self, samples, num_pois, neg_samples, rng, hard_neg_pool=None, hard_neg_ratio=0.0):
        self.samples = samples
        self.num_pois = num_pois
        self.neg = neg_samples
        self.rng = rng
        # C1 语义 hard-negative mining：hard_neg_pool 为 [num_pois, topk] 的语义近邻索引矩阵；
        # 仅当 pool 与 num_pois 索引对齐且 hard_neg_ratio>0 时启用，否则退回纯随机负样本。
        self.hard_neg_pool = hard_neg_pool
        self.hard_neg_ratio = hard_neg_ratio if (hard_neg_pool is not None and hard_neg_ratio > 0) else 0.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, hist, target = self.samples[idx]
        hist_set = set(hist)
        negs = set()
        # 1) 语义 hard 负样本（来自 target 的 bge 语义近邻池）——逼模型分离近似重复 POI
        n_hard = int(round(self.neg * self.hard_neg_ratio)) if self.hard_neg_ratio > 0 else 0
        if n_hard > 0:
            pool_list = [int(c) for c in self.hard_neg_pool[target]
                         if c != target and c not in hist_set]
            self.rng.shuffle(pool_list)  # 确定性（依赖共享 rng）
            for c in pool_list:
                if len(negs) >= n_hard:
                    break
                negs.add(c)
            # 近邻池不足 n_hard 时用随机补齐
            while len(negs) < n_hard:
                n = self.rng.randint(0, self.num_pois - 1)
                if n != target and n not in hist_set and n not in negs:
                    negs.add(n)
        # 2) 随机负样本（其余位置；保留部分易负样本维持 CE 全局分数校准）
        while len(negs) < self.neg:
            n = self.rng.randint(0, self.num_pois - 1)
            if n != target and n not in hist_set and n not in negs:
                negs.add(n)
        cands = [target] + list(negs)  # 正样本置于 index 0
        time_bins = [self.rng.randint(0, 24 * 7 - 1) for _ in hist]
        return uid, hist, time_bins, cands, 0, target  # 末位为真實下一 POI id（供全候选排名）


def _collate(batch):
    maxl = max(len(h) for uid, h, _, _, _, _ in batch)
    H, T, C, U, labels, true_tgt = [], [], [], [], [], []
    for uid, h, tb, c, lbl, tgt in batch:
        pad = [-1] * (maxl - len(h))
        H.append(h + pad)
        T.append(tb + [0] * (maxl - len(h)))
        C.append(c)
        U.append(uid)
        labels.append(lbl)
        true_tgt.append(tgt)
    return (torch.tensor(H), torch.tensor(T), torch.tensor(C),
            torch.tensor(labels), torch.tensor(U), torch.tensor(true_tgt))


def prepare_splits(checkins, seq_len, seed=42, train_ratio=0.8):
    """时序 + 用户级防泄漏划分：整用户归入训练或验证，避免序列信息泄漏。
    返回 (train_samples, val_samples, num_pois)，其中 sample = (user_id, hist_list, target)。"""
    rng = random.Random(seed)
    num_pois = max(p for _, seq in checkins for p in seq) + 1
    users = list({u for u, _ in checkins})
    rng.shuffle(users)
    n_train = int(len(users) * train_ratio)
    train_users = set(users[:n_train])
    val_users = set(users[n_train:])
    train_samples = _build_samples(checkins, seq_len, train_users)
    val_samples = _build_samples(checkins, seq_len, val_users)
    return train_samples, val_samples, num_pois


def train_model(cfg: Config, pois, checkins, device=None):
    device = device or cfg.device
    rng = random.Random(cfg.seed)
    # ---- 1. 构建旅游知识图谱 (C1) ----
    kg = TourismKG(cfg, LLMInterface()).build(pois, checkins)
    print("[KG] 边统计:", kg.stats())

    num_pois = len(pois)
    num_cats = kg.num_cats
    cat_ids = kg.cat_ids
    sem_vecs = kg.sem_vecs
    edge_index = kg.edge_index
    n_users = max(u for u, _ in checkins) + 1

    model = STKGNet(cfg, num_pois, num_cats, cat_ids, sem_vecs, edge_index, n_users=n_users).to(device)

    # ---- 2. 按用户拆分（防泄漏） ----
    train_samples, val_samples, _ = prepare_splits(checkins, cfg.seq_len, cfg.seed)

    train_ds = TrajDataset(train_samples, num_pois, cfg.neg_samples, rng)
    val_ds = TrajDataset(val_samples, num_pois, cfg.neg_samples, rng)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # ---- 3. 训练 ----
    for ep in range(cfg.epochs):
        model.train()
        total = 0.0
        for H, T, C, Y, U, _ in train_dl:
            H, T, C, Y, U = H.to(device), T.to(device), C.to(device), Y.to(device), U.to(device)
            scores = model(H, T, C, U)
            loss = loss_fn(scores, Y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        # ---- 4. 验证（全候选打分） ----
        model.eval()
        all_scores, all_tgt = [], []
        with torch.no_grad():
            for H, T, C, Y, U, TGT in val_dl:
                H, T, U = H.to(device), T.to(device), U.to(device)
                cand = torch.arange(num_pois).unsqueeze(0).expand(H.size(0), -1).to(device)
                sc = model(H, T, cand, U)
                all_scores.append(sc.cpu())
                all_tgt.append(TGT)
        metrics = rank_metrics(torch.cat(all_scores), torch.cat(all_tgt), k_list=(5, 10))
        print(f"[Epoch {ep+1:02d}] loss={total/len(train_dl):.4f} | "
              f"Val Recall@5={metrics['Recall@5']} NDCG@5={metrics['NDCG@5']} "
              f"Recall@10={metrics['Recall@10']} NDCG@10={metrics['NDCG@10']}")

    return model, metrics


def eval_ours_full(model, samples, num_pois, cfg, device=None, mask_hist=False):
    """对给定样本做全候选排名评估（复用已训练模型）。

    mask_hist: 无重复消费域（MovieLens/Steam/Gowalla，测试端 revisit_ratio≈0）须置 True，
               把历史已交互物品从候选中剔除——这是 SASRec 一系工作的标准协议。
               Foursquare-NYC 这类重访主导域必须保持 False，否则会屏蔽掉正确答案本身。
               开关由调用方按数据集 revisit_ratio 决定，且必须对 ours 与全部基线一视同仁。
    """
    device = device or cfg.device
    if not samples:
        return {"Recall@5": 0.0, "NDCG@5": 0.0, "Recall@10": 0.0, "NDCG@10": 0.0}
    rng = random.Random(cfg.seed)
    ds = TrajDataset(samples, num_pois, cfg.neg_samples, rng)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate)
    model.eval()
    all_scores, all_tgt = [], []
    with torch.no_grad():
        for H, T, C, Y, U, TGT in dl:
            H, T, U = H.to(device), T.to(device), U.to(device)
            cand = torch.arange(num_pois).unsqueeze(0).expand(H.size(0), -1).to(device)
            sc = model(H, T, cand, U)
            all_scores.append(sc.cpu())
            all_tgt.append(TGT)
    S = torch.cat(all_scores)
    Y = torch.cat(all_tgt)
    if mask_hist:
        # DataLoader shuffle=False，S 的行与 samples 一一对应
        assert S.size(0) == len(samples), f"行数不匹配: {S.size(0)} vs {len(samples)}"
        S = mask_history(S, samples, num_pois)
    metrics = rank_metrics(S, Y, k_list=(5, 10))
    metrics["__rank_diag__"] = rank_diag(S, Y)
    return metrics
