"""两阶段「语义无关重排」—— 在真正最强模型 ours (LLM-STKG) 之上改进。

动机：前序实验证明
  (1) BGE/LLM 语义作为「特征」注入（节点 init / resid / 重排特征）在本基准上 4 次证伪、
      正交；
  (2) 语义无关的两阶段重排器在 LightGCN 骨干上能补 NDCG（0.4499→0.4921，R@10+4.2、
      N@10+11.4%），但那主要是追回一个【弱】stage-1，未真正超越单模型。

本脚本把同一「语义无关两阶段重排」叠到【真正最强模型 ours (R@10=0.5612)】上，
验证它能否闭合 ours 的 NDCG 缺口（ours N@10≈0.256 vs LightGCN 0.271）。

阶段1 = 生产 ours（C1 bge + SGCP + C6 ctx + dot，seed42 权重，R@10=0.5612），冻结；
阶段2 = 独立小 MLP 重排器（输入 = [q ‖ emb[c] ‖ s1[c]]，不含 BGE），在 stage-1 top-K
        检索集内以「真目标排首位」的 CE 训练；推理用分数融合 final = s1 + λ·z(重排分)。

只跑语义无关路径（use_sem=False）——因前序实验已 4 次证明语义特征加 0 净增益。

输出 two_stage_ours.json（ours 仅 / ours+重排 + baseline_ranks.json LightGCN 对照）。
"""
import argparse
import json
import os
import sys
import time
import random
from collections import Counter

# ---- 环境铁律：6 线程前缀，避免 torch 偶发 segfault ----
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_stkg.config import Config                               # noqa: E402
from llm_stkg.data.foursquare_loader import load_real_nyc       # noqa: E402
from llm_stkg.train import _build_samples, eval_ours_full       # noqa: E402
from llm_stkg.evaluate import rank_metrics, rank_diag           # noqa: E402
from llm_stkg.model.stkg_net import STKGNet                     # noqa: E402
from llm_stkg.head_to_head import (build_kg, build_ui_edge,     # noqa: E402
                                    build_pop_prior)


# ----------------------------------------------------------------------
class ReRanker(nn.Module):
    """轻量重排器（语义无关版）：输入 = [q(64) ‖ emb[c](64) ‖ s1[c](1)]。"""

    def __init__(self, dim=64, hidden=64):
        super().__init__()
        in_dim = dim * 2 + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, q, item_emb, s1):
        # q:[B,K,dim] item_emb:[B,K,dim] s1:[B,K]
        x = torch.cat([q.float(), item_emb.float(), s1.float().unsqueeze(-1)], -1)
        return self.net(x).squeeze(-1)                          # [B, K]


class RerankDataset:
    """逐样本 (q, 候选列表特征)；位置 0 = 真目标。分块存 top-K 索引与 s1 值，省内存。"""

    def __init__(self, Q, emb, topk_idx, s1_topk, s1_tgt, targets, K):
        self.Q = Q
        self.emb = emb
        self.topk = topk_idx
        self.s1_topk = s1_topk
        self.s1_tgt = s1_tgt
        self.tgts = targets
        self.K = K

    def __len__(self):
        return len(self.tgts)

    def __getitem__(self, i):
        tgt = int(self.tgts[i])
        q = self.Q[i]
        tk = self.topk[i]
        mask = tk != tgt
        negs = tk[mask][:self.K - 1]
        lst = torch.cat([torch.tensor([tgt]), negs])            # [K]
        item = self.emb[lst]                                    # [K, dim]
        s1v = torch.cat([self.s1_tgt[i:i + 1],
                         self.s1_topk[i][mask][:self.K - 1]])    # [K]
        return q, item, s1v

    @staticmethod
    def collate(batch):
        q = torch.stack([b[0] for b in batch])
        item = torch.stack([b[1] for b in batch])
        s1 = torch.stack([b[2] for b in batch])
        return q, item, s1


# ----------------------------------------------------------------------
def build_ours_model(cfg, pois, checkins, num_pois, n_users, device, weight_path):
    """用生产配置构造 STKGNet 并加载 seed42 权重（R@10=0.5612）。"""
    kg = build_kg(cfg, pois, checkins)
    ui_edge = build_ui_edge(checkins, num_pois) if getattr(
        cfg, "use_ui_graph", True) else torch.empty(2, 0, dtype=torch.long)
    model = STKGNet(cfg, num_pois, kg.num_cats, kg.cat_ids, kg.sem_vecs,
                    kg.edge_index, n_users=n_users, user_item_edge=ui_edge,
                    pop_prior=build_pop_prior(checkins, num_pois),
                    cooc_matrix=None).to(device)
    print(f"[ours] 参数={sum(p.numel() for p in model.parameters())}，"
          f"加载权重: {weight_path}")
    sd = torch.load(weight_path, map_location=device)
    miss, unexp = model.load_state_dict(sd, strict=True)
    if miss or unexp:
        print(f"  ⚠️ load_state_dict 不匹配: missing={miss} unexpected={unexp}")
    model.eval()
    return model, kg


def ours_scores_Q(model, emb, samples, num_pois, device, batch=512):
    """返回 (S[n,N], Q[n,dim])：S=ours 全候选打分（含 C6 先验），Q=历史均值 POI 表征。"""
    model.eval()
    rng = random.Random(42)
    S, Q = [], []
    n = len(samples)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        bs = samples[s:e]
        maxl = max(len(h) for _, h, _ in bs)
        H, T, U = [], [], []
        for uid, h, _ in bs:
            H.append(h + [-1] * (maxl - len(h)))
            T.append([rng.randint(0, 24 * 7 - 1) for _ in h]
                     + [0] * (maxl - len(h)))
            U.append(uid)
        H = torch.tensor(H).to(device)
        T = torch.tensor(T).to(device)
        U = torch.tensor(U).to(device)
        B = H.size(0)
        cand = torch.arange(num_pois, device=device).unsqueeze(0).expand(B, -1)
        with torch.no_grad():
            sc = model(H, T, cand, U)                            # [B, N]
            mask = (H >= 0).float()
            clamped = H.clamp_min(0)
            ph = emb.to(device)[clamped]                         # [B,T,dim]
            q = ((ph * mask.unsqueeze(-1)).sum(1)
                 / mask.sum(1, keepdim=True).clamp_min(1.0))      # [B,dim]
        S.append(sc.cpu())
        Q.append(q.cpu())
    return torch.cat(S), torch.cat(Q)


def ours_train_topk(model, emb, train_samples, num_pois, tgt_train,
                    device, K, batch=512):
    """分块算训练集 stage-1 top-K 索引 + s1 值 + 目标 s1 值（避免 [n,N] 巨矩阵）。"""
    model.eval()
    rng = random.Random(42)
    topk_list, s1topk_list, s1tgt_list = [], [], []
    n = len(train_samples)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        bs = train_samples[s:e]
        maxl = max(len(h) for _, h, _ in bs)
        H, T, U = [], [], []
        for uid, h, _ in bs:
            H.append(h + [-1] * (maxl - len(h)))
            T.append([rng.randint(0, 24 * 7 - 1) for _ in h]
                     + [0] * (maxl - len(h)))
            U.append(uid)
        H = torch.tensor(H).to(device)
        T = torch.tensor(T).to(device)
        U = torch.tensor(U).to(device)
        B = H.size(0)
        cand = torch.arange(num_pois, device=device).unsqueeze(0).expand(B, -1)
        with torch.no_grad():
            sc = model(H, T, cand, U)                            # [B, N]
            tk = sc.topk(K).indices                             # [B, K]
            tv = sc.gather(1, tk)
            s1tgt = sc[torch.arange(B), tgt_train[s:e]]
        topk_list.append(tk.cpu())
        s1topk_list.append(tv.cpu())
        s1tgt_list.append(s1tgt.cpu())
    return (torch.cat(topk_list), torch.cat(s1topk_list),
            torch.cat(s1tgt_list))


def rerank_scores(reranker, emb, Q, tgts, s1_all, K, device, lam=0.3):
    """两阶段打分：final = s1 + λ·z(重排分)。返回 [n_test, N]。"""
    reranker.eval()
    finals = []
    with torch.no_grad():
        for i in range(len(tgts)):
            s1 = s1_all[i]
            lst = s1.topk(K).indices                             # 实际检索集
            item = emb[lst].to(device)
            s1v = s1[lst].to(device)
            qe = Q[i].unsqueeze(0).expand(len(lst), -1).to(device)
            logits = reranker(qe, item, s1v)                     # [K]
            z = (logits - logits.mean()) / (logits.std() + 1e-8)
            final = s1.clone()
            final[lst] = s1[lst] + lam * z.cpu()
            finals.append(final)
    return torch.stack(finals)


# ----------------------------------------------------------------------
def production_cfg(seed, device):
    """复现 production ours (c6_full_s42)：C1 bge + SGCP + C6 ctx + dot + user/200。"""
    cfg = Config()
    cfg.seed = seed
    cfg.device = device
    cfg.epochs = 30
    cfg.max_degree = 10
    cfg.batch_size = 1024
    cfg.lr = 4e-3
    cfg.use_bge = True
    cfg.bge_model_dir = "bge_model"
    cfg.bge_cache = "poi_bge_emb.npy"
    cfg.sem_dim = 768
    cfg.semantic_sim_thr = 0.90
    cfg.scorer = "dot"
    cfg.session_pool = "mean"
    cfg.use_sgcp = True
    cfg.hist_mode = "user"
    cfg.seq_len = 200
    cfg.prior_channels = "cnt,rec,pop"
    cfg.gate_mode = "context"
    # 默认即 True：use_user_pref / use_ui_graph / no_graph=False / homo_gnn=False /
    # use_residual=True / use_kg_channel=True / covisit_score=raw / cooc_agg=max
    cfg.num_pois = 0  # 由加载器覆盖
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--rerank_epochs", type=int, default=30)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--out", default="two_stage_ours.json")
    ap.add_argument("--weight", default="_c6u_seed42.pt")
    ap.add_argument("--baseline_json", default="baseline_ranks.json")
    ap.add_argument("--processed_dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.processed_dir is None:
        _here = os.path.dirname(os.path.abspath(__file__))
        for _rel in ("../data/real_foursquare_nyc/processed",
                     "../../data/real_foursquare_nyc/processed"):
            _c = os.path.normpath(os.path.join(_here, _rel))
            if os.path.exists(os.path.join(_c, "poi_meta.json")):
                args.processed_dir = _c
                break
    print(f"[data] processed_dir={args.processed_dir}", flush=True)

    torch.manual_seed(args.seed)
    pois, checkins, test_samples, num_pois, stats, _ = load_real_nyc(
        args.processed_dir, 0.0)
    users = list({u for u, _ in checkins})
    n_users = max(users) + 1
    train_samples = _build_samples(checkins, args.seq_len, set(users),
                                   hist_mode="user")
    print(f"[data] num_pois={num_pois} n_users={n_users} "
          f"n_test={len(test_samples)} n_train={len(train_samples)}")

    tgts = torch.tensor([int(t) for _, _, t in test_samples], dtype=torch.long)

    # 冷启动子集：训练频次 <= 5（与 production head_to_head_c1_sgcp.json 一致）
    freq = Counter(p for _, seq in checkins for p in seq)
    cold_set = {p for p in range(num_pois) if freq.get(p, 0) <= 5}

    if args.smoke:
        train_samples = train_samples[:2000]
        test_samples = test_samples[:200]
        tgts = tgts[:200]
        args.rerank_epochs = 1
        args.device = "cpu"
        args.K = 20
        print("[smoke] reduced train/test, cpu, 1 epoch, K=20")

    cold_idx = [i for i, (_, _, t) in enumerate(test_samples) if int(t) in cold_set]
    print(f"[cold] freq<=5 POIs={len(cold_set)} test_samples_in_cold={len(cold_idx)}")

    # ============ 阶段1：生产 ours（加载权重）============
    cfg = production_cfg(args.seed, args.device)
    cfg.num_pois = num_pois
    model, _ = build_ours_model(cfg, pois, checkins, num_pois, n_users,
                                 args.device, args.weight)

    # 用 production eval 复现 ours 基线（确认权重正确）
    print("\n=== Stage-1: LLM-STKG (ours), 生产评估 ===", flush=True)
    ours_base = eval_ours_full(model, test_samples, num_pois, cfg, args.device,
                               mask_hist=False)
    ours_base_cold = eval_ours_full(
        model, [s for i, s in enumerate(test_samples) if i in set(cold_idx)],
        num_pois, cfg, args.device, mask_hist=False) if cold_idx else None
    print(f"[ours] full   = {ours_base}")
    if ours_base_cold:
        print(f"[ours] cold(n={len(cold_idx)}) = {ours_base_cold}")

    # ============ 提取 emb / Q / s1 ============
    print("\n=== 提取 emb / Q / s1 ===", flush=True)
    with torch.no_grad():
        emb = model._get_poi_repr().detach().cpu()               # [N, hidden]
    t0 = time.time()
    S1_train_topk, s1_topk, s1_tgt = ours_train_topk(
        model, emb, train_samples, num_pois,
        torch.tensor([int(t) for _, _, t in train_samples], dtype=torch.long),
        args.device, args.K)
    print(f"[extract] train top-K 用时 {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    S1_test, Q_test = ours_scores_Q(model, emb, test_samples, num_pois, args.device)
    print(f"[extract] test s1/Q 用时 {time.time()-t0:.1f}s", flush=True)
    res_s1 = rank_metrics(S1_test, tgts)
    diag_s1 = rank_diag(S1_test, tgts)
    res_s1_cold = (rank_metrics(S1_test[cold_idx], tgts[cold_idx])
                   if cold_idx else None)
    print(f"[Stage-1 rerank-s1] full   = {res_s1}")
    if res_s1_cold:
        print(f"[Stage-1 rerank-s1] cold(n={len(cold_idx)}) = {res_s1_cold}")

    # ============ 阶段2：语义无关重排器 ============
    print("\n=== Stage-2: ReRanker (-Semantics, 语义无关) ===", flush=True)
    torch.manual_seed(args.seed)
    reranker = ReRanker(dim=64, hidden=64).to(args.device)
    opt = torch.optim.Adam(reranker.parameters(), lr=1e-3)
    ds = RerankDataset(Q_test[:0], emb, S1_train_topk, s1_topk, s1_tgt,
                       torch.tensor([int(t) for _, _, t in train_samples],
                                    dtype=torch.long), args.K)
    # 注意：RerankDataset 的 Q 用的是训练样本的 Q；我们上面没算 Q_train（仅算 Q_test）。
    # 重排器训练需要 Q_train；补算。
    print("[rerank] 计算 Q_train ...", flush=True)
    _, Q_train = ours_scores_Q(model, emb, train_samples, num_pois, args.device)
    ds.Q = Q_train
    loader = torch.utils.data.DataLoader(
        ds, batch_size=512, shuffle=True,
        collate_fn=RerankDataset.collate, num_workers=0)
    t0 = time.time()
    for ep in range(args.rerank_epochs):
        reranker.train()
        tot = 0.0
        for q, item, s1 in loader:
            q = q.to(args.device); item = item.to(args.device); s1 = s1.to(args.device)
            qe = q.unsqueeze(1).expand_as(item)
            logits = reranker(qe, item, s1)
            loss = nn.functional.cross_entropy(
                logits, torch.zeros(logits.size(0), dtype=torch.long,
                                    device=args.device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  [ReRanker] epoch {ep+1}/{args.rerank_epochs} "
                  f"loss={tot/len(loader):.4f}", flush=True)
    finals = rerank_scores(reranker, emb, Q_test, tgts, S1_test,
                           args.K, args.device, lam=args.lam)
    res = rank_metrics(finals, tgts)
    diag = rank_diag(finals, tgts)
    res_cold = (rank_metrics(finals[cold_idx], tgts[cold_idx])
                if cold_idx else None)
    print(f"[ReRanker (-Semantics)] full   = {res}")
    if res_cold:
        print(f"[ReRanker (-Semantics)] cold(n={len(cold_idx)}) = {res_cold}")
    print(f"[ReRanker (-Semantics)] time={time.time()-t0:.1f}s", flush=True)

    # ============ 对照表 + 落盘 ============
    results, diags = {}, {}
    results["LLM-STKG (ours) [production eval]"] = ours_base
    diags["LLM-STKG (ours) [production eval]"] = {
        "full": ours_base.get("__rank_diag__", {}),
        "cold": (ours_base_cold.get("__rank_diag__", {}) if ours_base_cold else None),
        "n_cold": len(cold_idx)}
    results["Ours + ReRanker (-Semantics)"] = res
    diags["Ours + ReRanker (-Semantics)"] = {
        "full": diag, "cold": res_cold, "n_cold": len(cold_idx)}

    if not args.smoke and os.path.exists(args.baseline_json):
        base = json.load(open(args.baseline_json, encoding="utf-8"))
        bl = base.get("results", {})
        for k in ("LightGCN", "BPR-MF", "Popularity (Pop)"):
            if k in bl:
                results[k] = bl[k]

    json.dump({
        "dataset": "Foursquare-NYC (LLM4POI, real)",
        "num_test": len(test_samples), "num_pois": num_pois,
        "rerank_epochs": args.rerank_epochs, "K": args.K, "seed": args.seed,
        "lambda": args.lam,
        "config": "production ours (C1 bge + SGCP + C6 ctx + dot, seed42) + "
                  "semantics-free reranker (use_sem=False)",
        "stats": stats,
        "results": results, "rank_diag": diags,
    }, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n=== 对照表（full-candidate）===")
    print(f"{'model':40s} {'R@5':>7s} {'R@10':>7s} {'N@10':>7s}")
    order = ["Popularity (Pop)", "BPR-MF", "LightGCN",
             "LLM-STKG (ours) [production eval]",
             "Ours + ReRanker (-Semantics)"]
    for k in order:
        if k in results:
            r = results[k]
            print(f"{k:40s} {r['Recall@5']:7.4f} {r['Recall@10']:7.4f} "
                  f"{r['NDCG@10']:7.4f}")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
