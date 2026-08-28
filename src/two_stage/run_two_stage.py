"""两阶段「LLM 语义重排」—— 在现有最强骨干 (LightGCN) 之上改进。

动机（判废 v2 + 嵌入注入之后）：
  在本基准上 BGE/LLM 语义与共现信号正交——把语义"塞进物品嵌入"（resid/init/freeze）
  会塌缩，因为污染了已学好的共现几何。但语义在"区分功能近似的相似 POI"（两家相邻咖啡店）
  上有真实判别力，而这恰恰是 top 精度 / NDCG 的短板。

方法（与失败模式本质不同）：
  - 阶段1（不变，冻结）：LightGCN 负责全候选检索，产出校准的全排名分数 s1 与物品嵌入 emb。
  - 阶段2（新增，独立）：对阶段1 召回的 top-K 候选，用一个轻量重排器
        score = MLP( [会话表征 q ‖ 候选嵌入 emb[c] ‖ 冻结BGE投影 bge[c] ‖ 阶段1分数 s1[c]] )
    以"在检索集内把真目标排到首位"的交叉熵训练。语义只作为【重排特征】进入独立小模型，
    绝不混入共现嵌入空间、不破坏阶段1的全排名校准、不替换负样本分布——避开了此前
    所有失败模式的共同根因。

新颖性：LLM 桥接以"重排依据"形式保留（区别于纯随机/流行度重排），且重排器在检索后
独立训练，架构隔离。

消融（同脚本一次跑完）：
  (A) LightGCN（阶段1 仅）              —— baseline
  (B) 重排器 - 语义（输入不含 bge）      —— 测试"重排器本身"是否有用
  (C) 重排器 + 语义（输入含 bge）        —— 提案；若 C>B>A 则语义经重排真能补 NDCG

输出 two_stage.json（含 baseline_ranks.json 的 LightGCN 对照 + 各路 full/cold 指标与 rank_diag）。
"""
import argparse
import json
import os
import sys
import time
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

from llm_stkg.data.foursquare_loader import load_real_nyc   # noqa: E402
from llm_stkg.train import _build_samples                   # noqa: E402
from llm_stkg.evaluate import rank_metrics, rank_diag       # noqa: E402
from llm_stkg.baselines import LightGCN as VanillaLightGCN  # noqa: E402


# ----------------------------------------------------------------------
class ReRanker(nn.Module):
    """轻量重排器：输入候选级特征，输出标量重排分数。

    use_sem=True  : 输入 = [q(64) ‖ emb[c](64) ‖ s1[c](1) ‖ bge[c](64)]  (提案)
    use_sem=False : 输入 = [q(64) ‖ emb[c](64) ‖ s1[c](1)]            (消融)
    """

    def __init__(self, dim=64, use_sem=True, hidden=64):
        super().__init__()
        self.use_sem = use_sem
        in_dim = dim * 2 + 1 + (dim if use_sem else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, q, item_emb, s1, bge=None):
        # q:[B,K,dim]（已广播） item_emb:[B,K,dim] s1:[B,K] bge:[B,K,dim]
        parts = [q.float(), item_emb.float(), s1.float().unsqueeze(-1)]
        if self.use_sem and bge is not None:
            parts.append(bge.float())
        x = torch.cat(parts, -1)
        return self.net(x).squeeze(-1)            # [B, K]


class RerankDataset:
    """逐样本构造 (q, 候选列表特征)；列表长度固定 = K，位置 0 = 真目标。

    s1 特征不存全量 [n,N] 矩阵（1.6GB），改存分块算出的 top-K 索引 s1_topk
    与真目标 s1 值 s1_tgt，按列表即时拼装，峰值内存 ~32MB。
    """

    def __init__(self, Q, emb, bge_proj, topk_idx, s1_topk, s1_tgt, targets, K):
        self.Q = Q
        self.emb = emb
        self.bge = bge_proj
        self.topk = topk_idx
        self.s1_topk = s1_topk
        self.s1_tgt = s1_tgt
        self.tgts = targets
        self.K = K

    def __len__(self):
        return len(self.tgts)

    def __getitem__(self, i):
        tgt = int(self.tgts[i])
        q = self.Q[i]                                   # [dim]
        tk = self.topk[i]                               # [K]
        mask = tk != tgt
        negs = tk[mask][:self.K - 1]                    # 至多 K-1 个负样本
        lst = torch.cat([torch.tensor([tgt]), negs])    # [K]（长度恒为 K）
        item = self.emb[lst]                            # [K, dim]
        bge = self.bge[lst]                             # [K, dim]
        s1v = torch.cat([self.s1_tgt[i:i + 1],
                         self.s1_topk[i][mask][:self.K - 1]])  # [K]
        return q, item, s1v, bge


def collate(batch):
    q = torch.stack([b[0] for b in batch])              # [B, dim]
    item = torch.stack([b[1] for b in batch])           # [B, K, dim]
    s1 = torch.stack([b[2] for b in batch])             # [B, K]
    bge = torch.stack([b[3] for b in batch])            # [B, K, dim]
    return q, item, s1, bge


# ----------------------------------------------------------------------
def build_stage1(train_samples, test_samples, num_pois, epochs, lr, bs,
                 device, seed):
    """训练并冻结 LightGCN，返回 emb[N,dim](cpu) 与 (Q_train, Q_test)。"""
    torch.manual_seed(seed)
    model = VanillaLightGCN(num_pois, dim=64, n_layers=3)
    model.fit(train_samples, epochs=epochs, lr=lr, bs=bs, device=device)
    emb = model._emb().detach().cpu()                  # [N, dim] 冻结
    # 预计算所有样本的会话表征（= 历史物品嵌入均值，与 session_predict 一致）
    def query_matrix(samples):
        rows = []
        for _, h, _ in samples:
            if not h:
                h = [0]
            hv = emb[torch.tensor(h, dtype=torch.long)].mean(0)
            rows.append(hv)
        return torch.stack(rows)                        # [n, dim]
    Q_train = query_matrix(train_samples)
    Q_test = query_matrix(test_samples)
    return emb, Q_train, Q_test


def rerank_scores(reranker, emb, bge_proj, Q, tgts, s1_all, K, device,
                  use_sem=True, lam=0.3):
    """对测试集做两阶段打分：阶段1 s1 + 阶段2 重排块置于榜首。返回 [n_test, N]。"""
    reranker.eval()
    finals = []
    with torch.no_grad():
        for i in range(len(tgts)):
            q = Q[i]                                    # [dim]
            s1 = s1_all[i]                              # [N]
            # 诚实协议：只重排阶段1 实际召回的 top-K；target 在块内才被重排，
            # 不在块内则保持阶段1 原排名（重排器无法凭空召回）。
            lst = s1.topk(K).indices                    # [K] 实际检索集
            item = emb[lst].to(device)                  # [K, dim]
            bge = bge_proj[lst].to(device)              # [K, dim]
            s1v = s1[lst].to(device)                    # [K]
            qe = q.unsqueeze(0).expand(len(lst), -1).to(device)  # [K, dim]
            logits = reranker(qe, item, s1v, bge if use_sem else None)  # [K]
            # 分数融合（非替换）：final = s1 + λ·z(重排分)。
            # 重排器只作对 stage-1 的微扰/锐化——弱时 final≈s1（不伤），
            # 强时把真目标相对其 CF 近邻抬升，改善头部排序（NDCG/R@10）。
            z = (logits - logits.mean()) / (logits.std() + 1e-8)
            final = s1.clone()
            final[lst] = s1[lst] + lam * z.cpu()
            finals.append(final)
    return torch.stack(finals)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--K", type=int, default=50,
                    help="重排检索块大小（阶段1 召回 top-K 再精排）")
    ap.add_argument("--rerank_epochs", type=int, default=30)
    ap.add_argument("--lam", type=float, default=0.3,
                    help="分数融合权重：final = s1 + lam·z(重排分)")
    ap.add_argument("--out", default="two_stage.json")
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
    sem = np.load("poi_bge_emb.npy")
    assert sem.shape[0] == num_pois, f"sem {sem.shape} vs num_pois {num_pois}"

    hl = sum(len(h) for _, h, _ in train_samples) / max(1, len(train_samples))
    rv = sum(1 for _, h, t in train_samples if t in set(h)) / max(1, len(train_samples))
    print(f"[data] num_pois={num_pois} n_users={n_users} "
          f"n_test={len(test_samples)} n_train={len(train_samples)}")
    print(f"[samples] hist_mode=user seq_len={args.seq_len} "
          f"hist_len_mean={hl:.1f} revisit_ratio={rv:.4f}")

    tgts = torch.tensor([int(t) for _, _, t in test_samples], dtype=torch.long)

    # 冷启动子集：训练频次 <= 2 的 POI 作为目标
    freq = Counter(p for _, seq in checkins for p in seq)
    cold_set = {p for p in range(num_pois) if freq.get(p, 0) <= 2}

    if args.smoke:
        train_samples = train_samples[:2000]
        test_samples = test_samples[:200]
        tgts = tgts[:200]
        args.epochs = 1
        args.rerank_epochs = 1
        args.device = "cpu"
        args.K = 20
        print("[smoke] reduced train/test, cpu, 1 epoch, K=20")

    cold_idx = [i for i, (_, _, t) in enumerate(test_samples) if int(t) in cold_set]
    print(f"[cold] freq<=2 POIs={len(cold_set)} test_samples_in_cold={len(cold_idx)}")

    # ============ 阶段1 ============
    print("\n=== Stage-1: LightGCN (frozen) ===", flush=True)
    t0 = time.time()
    emb, Q_train, Q_test = build_stage1(
        train_samples, test_samples, num_pois, args.epochs, 1e-3, 1024,
        args.device, args.seed)
    # 阶段1 全候选分数（= session_predict）
    with torch.no_grad():
        S1_test = (Q_test @ emb.T)                      # [n_test, N]
    res_s1 = rank_metrics(S1_test, tgts)
    diag_s1 = rank_diag(S1_test, tgts)
    res_s1_cold = (rank_metrics(S1_test[cold_idx], tgts[cold_idx])
                   if cold_idx else None)
    print(f"[Stage-1] full   = {res_s1}")
    if res_s1_cold:
        print(f"[Stage-1] cold(n={len(cold_idx)}) = {res_s1_cold}")
    print(f"[Stage-1] time={time.time()-t0:.1f}s", flush=True)

    # 预计算训练集阶段1 top-K（分块，避免 [n_train, N] 1.6GB 矩阵）
    tgt_train = torch.tensor([int(t) for _, _, t in train_samples],
                             dtype=torch.long)
    chunk = 10000
    topk_list, s1topk_list, s1tgt_list = [], [], []
    with torch.no_grad():
        for s in range(0, Q_train.size(0), chunk):
            e = min(s + chunk, Q_train.size(0))
            s1c = Q_train[s:e] @ emb.T                  # [c, N]
            tk = s1c.topk(args.K).indices
            tv = s1c.gather(1, tk)
            s1tgt = s1c[torch.arange(e - s), tgt_train[s:e]]  # [c] 真目标 s1
            topk_list.append(tk)
            s1topk_list.append(tv)
            s1tgt_list.append(s1tgt)
    topk_idx = torch.cat(topk_list)                    # [n_train, K]
    s1_topk = torch.cat(s1topk_list)                   # [n_train, K]
    s1_tgt = torch.cat(s1tgt_list)                     # [n_train]
    del topk_list, s1topk_list, s1tgt_list

    # 固定随机投影把 BGE 768→64（冻结，保留余弦结构，Johnson-Lindenstrauss）
    rng_np = np.random.RandomState(12345)
    W_bge = (rng_np.randn(sem.shape[1], 64).astype(np.float32)
             / np.sqrt(sem.shape[1]))
    bge_proj = torch.tensor(sem.astype(np.float32) @ W_bge).float()   # [N, 64] 冻结

    results, diags = {}, {}
    results["LightGCN (Stage-1 only)"] = res_s1
    diags["LightGCN (Stage-1 only)"] = {
        "full": diag_s1, "cold": res_s1_cold, "n_cold": len(cold_idx)}

    # ============ 阶段2：重排器（无语义 / 有语义）============
    for use_sem in (False, True):
        key = ("ReRanker (+Semantics)" if use_sem
               else "ReRanker (-Semantics)")
        print(f"\n=== Stage-2: {key} (K={args.K}) ===", flush=True)
        torch.manual_seed(args.seed)
        reranker = ReRanker(dim=64, use_sem=use_sem, hidden=64).to(args.device)
        opt = torch.optim.Adam(reranker.parameters(), lr=1e-3)
        ds = RerankDataset(Q_train, emb, bge_proj, topk_idx, s1_topk,
                           s1_tgt, tgt_train, args.K)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=512, shuffle=True, collate_fn=collate,
            num_workers=0)
        t0 = time.time()
        for ep in range(args.rerank_epochs):
            reranker.train()
            tot = 0.0
            for q, item, s1, bge in loader:
                q = q.to(args.device)
                item = item.to(args.device)
                s1 = s1.to(args.device)
                bge = bge.to(args.device)
                qe = q.unsqueeze(1).expand_as(item)    # [B, K, dim]
                logits = reranker(qe, item, s1, bge if use_sem else None)
                # 标签恒为 0：列表位置 0 = 真目标
                loss = nn.functional.cross_entropy(
                    logits, torch.zeros(logits.size(0), dtype=torch.long,
                                        device=args.device))
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  [{key}] epoch {ep+1}/{args.rerank_epochs} "
                      f"loss={tot/len(loader):.4f}", flush=True)
        # 推理：两阶段打分
        finals = rerank_scores(reranker, emb, bge_proj, Q_test, tgts,
                               S1_test, args.K, args.device, use_sem=use_sem,
                               lam=args.lam)
        res = rank_metrics(finals, tgts)
        diag = rank_diag(finals, tgts)
        res_cold = (rank_metrics(finals[cold_idx], tgts[cold_idx])
                    if cold_idx else None)
        results[key] = res
        diags[key] = {"full": diag, "cold": res_cold, "n_cold": len(cold_idx)}
        print(f"[{key}] full   = {res}")
        if res_cold:
            print(f"[{key}] cold(n={len(cold_idx)}) = {res_cold}")
        print(f"[{key}] time={time.time()-t0:.1f}s", flush=True)
        if args.smoke:
            continue

    # 载入基线 LightGCN 对照
    if not args.smoke and os.path.exists(args.baseline_json):
        base = json.load(open(args.baseline_json, encoding="utf-8"))
        bl = base.get("results", {})
        for k in ("LightGCN", "BPR-MF", "Popularity (Pop)"):
            if k in bl:
                results[k] = bl[k]
        diags["LightGCN"] = base.get("rank_diag", {}).get("full", {})

    _dump(args.out, num_pois, len(test_samples), train_samples, hl, rv,
          results, diags, args, stats)

    print("\n=== 对照表（full-candidate）===")
    print(f"{'model':30s} {'R@5':>7s} {'R@10':>7s} {'N@10':>7s}")
    order = ["Popularity (Pop)", "BPR-MF", "LightGCN",
             "LightGCN (Stage-1 only)", "ReRanker (-Semantics)",
             "ReRanker (+Semantics)"]
    for k in order:
        if k in results:
            r = results[k]
            print(f"{k:30s} {r['Recall@5']:7.4f} {r['Recall@10']:7.4f} "
                  f"{r['NDCG@10']:7.4f}")
    print(f"\n[saved] {args.out}")


def _dump(out, num_pois, n_test, train_samples, hl, rv, results, diags, args, stats):
    json.dump({
        "dataset": "Foursquare-NYC (LLM4POI, real)",
        "num_test": n_test, "num_pois": num_pois,
        "epochs": args.epochs, "rerank_epochs": args.rerank_epochs,
        "K": args.K, "seed": args.seed,
        "protocol": {"hist_mode": "user", "seq_len": args.seq_len,
                     "n_train_samples": len(train_samples),
                     "hist_len_mean": round(hl, 1), "revisit_ratio": round(rv, 4)},
        "stats": stats,
        "results": results, "rank_diag": diags,
    }, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
