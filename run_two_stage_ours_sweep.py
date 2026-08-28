"""多算例对照：生产 ours (LLM-STKG, C6) + 两阶段重排器，{变体} × {种子}。

动机（用户 17 日要求"多算几个算例对比"）：
  原 run_two_stage_ours.py 只跑了单一种子、单一语义无关配置，用作
  "两阶段重排是死胡同 / 增益噪声级" 的支撑太单薄。本脚本补算多组算例：
    (1) 多随机种子（42/123/777）证明语义无关重排叠加到强 stage-1(ours) 上
        的增益是噪声级（跨种子方差小）；
    (2) 在 ours 上补跑 +BGE 语义重排（拼 kg.sem_vecs 768 维），闭合
        "LLM 语义作为特征第 5 次证伪"——即便喂语义，强模型增益仍为 0。

阶段1 = 生产 ours（C1 bge + SGCP + C6 ctx + dot, seed42 权重），冻结一次，
        复用其 s1/Q/emb/train-topK/sem 给所有重排配置，避免重复前向。

输出 two_stage_ours_sweep.json（全量对照表）。
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
from run_two_stage_ours import production_cfg, build_ours_model  # noqa: E402

# ----------------------------------------------------------------------
class ReRanker(nn.Module):
    """灵活重排器：输入 = [q(dim) ‖ item(item_dim) ‖ s1(1)]。
    item_dim = 64（语义无关，仅 POI 表征）或 64+768（再拼 BGE 语义）。"""

    def __init__(self, dim=64, item_dim=64, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + item_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, q, item, s1):
        x = torch.cat([q.float(), item.float(), s1.float().unsqueeze(-1)], -1)
        return self.net(x).squeeze(-1)                          # [B, K]


class RerankDataset:
    """逐样本 (q, 候选列表特征, s1)；位置 0 = 真目标。
    item = emb[lst]（语义无关）或 cat([emb[lst], sem[lst]])（含 BGE）。"""

    def __init__(self, Q, emb, sem, topk_idx, s1_topk, s1_tgt, targets, K):
        self.Q = Q
        self.emb = emb
        self.sem = sem
        self.topk = topk_idx
        self.s1_topk = s1_topk
        self.s1_tgt = s1_tgt
        self.tgts = targets
        self.K = K

    def __len__(self):
        return len(self.tgts)

    def __getitem__(self, i):
        tgt = int(self.tgts[i])
        q = self.Q[i].unsqueeze(0).expand(self.K, -1)            # [K, dim]
        tk = self.topk[i]
        mask = tk != tgt
        negs = tk[mask][:self.K - 1]
        lst = torch.cat([torch.tensor([tgt]), negs])            # [K]
        item = self.emb[lst]                                    # [K, 64]
        if self.sem is not None:
            item = torch.cat([item, self.sem[lst]], -1)         # [K, 64+768]
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
def extract_stage1(model, emb, samples, num_pois, device, K,
                   batch=512, want_topk=True, want_Q=True):
    """单次前向，按需返回 (S, Q, topk, s1_topk, s1_tgt)。
    时间分箱固定 random.Random(42) → 跨算例确定性。"""
    model.eval()
    rng = random.Random(42)
    S_list, Q_list, topk_list, s1topk_list, s1tgt_list = [], [], [], [], []
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
            if want_topk:
                tk = sc.topk(K).indices
                tv = sc.gather(1, tk)
                tgt_idx = torch.tensor([int(t) for _, _, t in bs],
                                       dtype=torch.long, device=device)
                s1tgt = sc[torch.arange(B), tgt_idx]
                topk_list.append(tk.cpu())
                s1topk_list.append(tv.cpu())
                s1tgt_list.append(s1tgt.cpu())
        if want_Q:
            Q_list.append(q.cpu())
        S_list.append(sc.cpu())
    S = torch.cat(S_list)
    Q = torch.cat(Q_list) if want_Q else None
    if want_topk:
        return S, Q, torch.cat(topk_list), torch.cat(s1topk_list), torch.cat(s1tgt_list)
    return S, Q, None, None, None


def rerank_scores(reranker, emb, sem, Q, tgts, s1_all, K, device, lam=0.3):
    """两阶段打分：final = s1 + λ·z(重排分)。返回 [n_test, N]。"""
    reranker.eval()
    finals = []
    with torch.no_grad():
        for i in range(len(tgts)):
            s1 = s1_all[i]
            lst = s1.topk(K).indices                             # 实际检索集
            item = emb[lst].to(device)
            if sem is not None:
                item = torch.cat([item, sem[lst].to(device)], -1)
            s1v = s1[lst].to(device)
            qe = Q[i].unsqueeze(0).expand(len(lst), -1).to(device)
            logits = reranker(qe, item, s1v)                     # [K]
            z = (logits - logits.mean()) / (logits.std() + 1e-8)
            final = s1.clone()
            final[lst] = s1[lst] + lam * z.cpu()
            finals.append(final)
    return torch.stack(finals)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="42,123,777")
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--rerank_epochs", type=int, default=30)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", default="two_stage_ours_sweep.json")
    ap.add_argument("--weight", default="_c6u_seed42.pt")
    ap.add_argument("--baseline_json", default="baseline_ranks.json")
    ap.add_argument("--processed_dir", default=None)
    ap.add_argument("--variants", default="sem_free,with_bge")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    variants = [v for v in args.variants.split(",") if v in ("sem_free", "with_bge")]
    print(f"[cfg] seeds={seeds} variants={variants} K={args.K} "
          f"rerank_epochs={args.rerank_epochs} lam={args.lam} hidden={args.hidden}",
          flush=True)

    if args.processed_dir is None:
        _here = os.path.dirname(os.path.abspath(__file__))
        for _rel in ("../data/real_foursquare_nyc/processed",
                     "../../data/real_foursquare_nyc/processed"):
            _c = os.path.normpath(os.path.join(_here, _rel))
            if os.path.exists(os.path.join(_c, "poi_meta.json")):
                args.processed_dir = _c
                break
    print(f"[data] processed_dir={args.processed_dir}", flush=True)

    pois, checkins, test_samples, num_pois, stats, _ = load_real_nyc(
        args.processed_dir, 0.0)
    users = list({u for u, _ in checkins})
    n_users = max(users) + 1
    train_samples = _build_samples(checkins, 200, set(users), hist_mode="user")
    print(f"[data] num_pois={num_pois} n_users={n_users} "
          f"n_test={len(test_samples)} n_train={len(train_samples)}")

    tgts = torch.tensor([int(t) for _, _, t in test_samples], dtype=torch.long)
    train_tgts = torch.tensor([int(t) for _, _, t in train_samples],
                               dtype=torch.long)

    # 冷启动子集：训练频次 <= 5
    freq = Counter(p for _, seq in checkins for p in seq)
    cold_set = {p for p in range(num_pois) if freq.get(p, 0) <= 5}
    cold_idx = [i for i, (_, _, t) in enumerate(test_samples) if int(t) in cold_set]
    print(f"[cold] freq<=5 POIs={len(cold_set)} test_in_cold={len(cold_idx)}")

    if args.smoke:
        train_samples = train_samples[:2000]
        test_samples = test_samples[:200]
        tgts = tgts[:200]
        train_tgts = train_tgts[:2000]
        args.rerank_epochs = 1
        args.device = "cpu"
        args.K = 20
        cold_idx = [i for i in cold_idx if i < 200]
        print("[smoke] reduced train/test, cpu, 1 epoch, K=20")

    # ============ 阶段1：生产 ours（加载权重，冻结一次）============
    cfg = production_cfg(42, args.device)
    cfg.num_pois = num_pois
    model, kg = build_ours_model(cfg, pois, checkins, num_pois, n_users,
                                  args.device, args.weight)

    print("\n=== Stage-1: LLM-STKG (ours), 生产评估 ===", flush=True)
    ours_base = eval_ours_full(model, test_samples, num_pois, cfg, args.device,
                               mask_hist=False)
    ours_base_cold = eval_ours_full(
        model, [s for i, s in enumerate(test_samples) if i in set(cold_idx)],
        num_pois, cfg, args.device, mask_hist=False) if cold_idx else None
    print(f"[ours] full = {ours_base}")
    if ours_base_cold:
        print(f"[ours] cold(n={len(cold_idx)}) = {ours_base_cold}")

    # ============ 阶段1 特征提取（一次）============
    print("\n=== 提取 emb / Q / s1 / train-topK / sem（一次）===", flush=True)
    with torch.no_grad():
        emb = model._get_poi_repr().detach().cpu()               # [N, hidden=64]
    sem_all = None
    if "with_bge" in variants:
        sem_all = torch.as_tensor(
            np.array(kg.sem_vecs, dtype=np.float32)).cpu()       # [N, 768]
        print(f"[sem] BGE 语义矩阵 = {tuple(sem_all.shape)}")

    t0 = time.time()
    # extract_stage1 返回 (S, Q, topk, s1_topk, s1_tgt)
    _S_train, Q_train, S1_train_topk, s1_topk, s1_tgt = extract_stage1(
        model, emb, train_samples, num_pois, args.device,
        args.K, want_topk=True, want_Q=True)
    print(f"[extract] train top-K+Q 用时 {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    S1_test, Q_test, *_ = extract_stage1(
        model, emb, test_samples, num_pois, args.device, args.K,
        want_topk=False, want_Q=True)
    print(f"[extract] test s1/Q 用时 {time.time()-t0:.1f}s", flush=True)

    res_s1 = rank_metrics(S1_test, tgts)
    res_s1_cold = (rank_metrics(S1_test[cold_idx], tgts[cold_idx])
                   if cold_idx else None)
    print(f"[Stage-1 rerank-s1] full = {res_s1}")
    if res_s1_cold:
        print(f"[Stage-1 rerank-s1] cold(n={len(cold_idx)}) = {res_s1_cold}")

    # ============ 阶段2：多变体 × 多种子重排扫描 ============
    results, diags = {}, {}
    results["LLM-STKG (ours) [production eval]"] = ours_base
    diags["LLM-STKG (ours) [production eval]"] = {
        "full": ours_base.get("__rank_diag__", {}),
        "cold": (ours_base_cold.get("__rank_diag__", {}) if ours_base_cold else None),
        "n_cold": len(cold_idx)}
    results["Ours stage-1 (s1 only)"] = res_s1
    diags["Ours stage-1 (s1 only)"] = {
        "full": rank_diag(S1_test, tgts),
        "cold": (rank_diag(S1_test[cold_idx], tgts[cold_idx]) if cold_idx else None),
        "n_cold": len(cold_idx)}

    for variant in variants:
        use_sem = (variant == "with_bge")
        item_dim = 64 + (768 if use_sem else 0)
        sem = sem_all if use_sem else None
        vlabel = "ReRanker(+BGE)" if use_sem else "ReRanker(-Semantics)"
        print(f"\n=== Stage-2: {vlabel} (item_dim={item_dim}) ===", flush=True)
        ds = RerankDataset(Q_train, emb, sem, S1_train_topk, s1_topk, s1_tgt,
                           train_tgts, args.K)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=512, shuffle=True,
            collate_fn=RerankDataset.collate, num_workers=0)
        per_seed_r10, per_seed_n10 = [], []
        per_seed_r10_cold, per_seed_n10_cold = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            reranker = ReRanker(dim=64, item_dim=item_dim,
                                hidden=args.hidden).to(args.device)
            opt = torch.optim.Adam(reranker.parameters(), lr=1e-3)
            t0 = time.time()
            for ep in range(args.rerank_epochs):
                reranker.train()
                tot = 0.0
                for q, item, s1 in loader:
                    q = q.to(args.device); item = item.to(args.device)
                    s1 = s1.to(args.device)
                    logits = reranker(q, item, s1)
                    loss = nn.functional.cross_entropy(
                        logits, torch.zeros(logits.size(0), dtype=torch.long,
                                            device=args.device))
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += loss.item()
                if (ep + 1) % 5 == 0 or ep == 0:
                    print(f"  [{vlabel} s{seed}] epoch {ep+1}/"
                          f"{args.rerank_epochs} loss={tot/len(loader):.4f}",
                          flush=True)
            finals = rerank_scores(reranker, emb, sem, Q_test, tgts, S1_test,
                                   args.K, args.device, lam=args.lam)
            res = rank_metrics(finals, tgts)
            diag = rank_diag(finals, tgts)
            res_cold = (rank_metrics(finals[cold_idx], tgts[cold_idx])
                        if cold_idx else None)
            print(f"[{vlabel} s{seed}] full   = {res}")
            if res_cold:
                print(f"[{vlabel} s{seed}] cold(n={len(cold_idx)}) = {res_cold}")
            print(f"[{vlabel} s{seed}] time={time.time()-t0:.1f}s", flush=True)
            key = f"Ours + {vlabel} s{seed}"
            results[key] = res
            diags[key] = {"full": diag, "cold": res_cold, "n_cold": len(cold_idx),
                          "variant": variant, "seed": seed,
                          "item_dim": item_dim}
            per_seed_r10.append(res["Recall@10"])
            per_seed_n10.append(res["NDCG@10"])
            if res_cold:
                per_seed_r10_cold.append(res_cold["Recall@10"])
                per_seed_n10_cold.append(res_cold["NDCG@10"])
        # 跨种子聚合
        agg_key = f"Ours + {vlabel} [agg {len(seeds)} seeds]"
        results[agg_key] = {
            "Recall@5_mean": float(np.mean([results[f'Ours + {vlabel} s{s}']["Recall@5"] for s in seeds])),
            "Recall@10_mean": float(np.mean(per_seed_r10)),
            "NDCG@10_mean": float(np.mean(per_seed_n10)),
            "Recall@10_std": float(np.std(per_seed_r10)),
            "NDCG@10_std": float(np.std(per_seed_n10)),
            "Recall@10_min": float(np.min(per_seed_r10)),
            "Recall@10_max": float(np.max(per_seed_r10)),
            "NDCG@10_min": float(np.min(per_seed_n10)),
            "NDCG@10_max": float(np.max(per_seed_n10)),
            "Cold Recall@10_mean": (float(np.mean(per_seed_r10_cold))
                                    if per_seed_r10_cold else None),
            "Cold NDCG@10_mean": (float(np.mean(per_seed_n10_cold))
                                  if per_seed_n10_cold else None),
        }
        diags[agg_key] = {"variant": variant, "seeds": seeds,
                          "n_cold": len(cold_idx)}
        print(f"[{vlabel} agg] R@10={results[agg_key]['Recall@10_mean']:.4f}"
              f"±{results[agg_key]['Recall@10_std']:.4f} "
              f"N@10={results[agg_key]['NDCG@10_mean']:.4f}"
              f"±{results[agg_key]['NDCG@10_std']:.4f}")

    # ============ 对照表 + 落盘 ============
    if not args.smoke and os.path.exists(args.baseline_json):
        base = json.load(open(args.baseline_json, encoding="utf-8"))
        bl = base.get("results", {})
        for k in ("LightGCN", "BPR-MF", "Popularity (Pop)"):
            if k in bl:
                results[k] = bl[k]

    json.dump({
        "dataset": "Foursquare-NYC (LLM4POI, real)",
        "num_test": len(test_samples), "num_pois": num_pois,
        "rerank_epochs": args.rerank_epochs, "K": args.K, "seeds": seeds,
        "lambda": args.lam, "variants": variants, "hidden": args.hidden,
        "config": "production ours (C1 bge + SGCP + C6 ctx + dot, seed42) + "
                  "two-stage reranker sweep {sem_free, with_bge} x seeds",
        "stats": stats,
        "results": results, "rank_diag": diags,
    }, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n=== 对照表（full-candidate）===")
    print(f"{'model':46s} {'R@5':>7s} {'R@10':>7s} {'N@10':>7s}")
    order = ["Popularity (Pop)", "BPR-MF", "LightGCN",
             "LLM-STKG (ours) [production eval]", "Ours stage-1 (s1 only)"]
    for variant in variants:
        vlabel = "ReRanker(+BGE)" if variant == "with_bge" else "ReRanker(-Semantics)"
        order.append(f"Ours + {vlabel} [agg {len(seeds)} seeds]")
        for s in seeds:
            order.append(f"Ours + {vlabel} s{s}")
    for k in order:
        if k in results:
            r = results[k]
            if "Recall@10_mean" in r:
                print(f"{k:46s} {'':>7s} {r['Recall@10_mean']:7.4f}±"
                      f"{r['Recall@10_std']:.4f} {r['NDCG@10_mean']:7.4f}±"
                      f"{r['NDCG@10_std']:.4f}")
            else:
                print(f"{k:46s} {r['Recall@5']:7.4f} {r['Recall@10']:7.4f} "
                      f"{r['NDCG@10']:7.4f}")
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
