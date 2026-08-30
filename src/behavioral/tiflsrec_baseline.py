# -*- coding: utf-8 -*-
"""
TiIfSRec (recipe-level re-implementation) -- dual-gated time/frequency sequential
recommendation, used as a *learned behavioral-prior* baseline for the LLM-STKG paper.

Original: Wang, X. "Context-Aware and Adaptive Multi-Scale Interest Modeling for
Sequential Recommendation", uOttawa PhD thesis 2026 -- the TiIfSRec model: a dual-gated
recurrent architecture where (i) time-interval signals control the decay of historical
preferences and (ii) item-frequency signals regulate the direction of state updates to
mitigate popularity bias, plus an attention module that highlights informative history.

Why recipe-level (not the released code):
  The released TiIfSRec (github.com/xiaownwang/TiIfSRec) expects timestamped
  user-item sequences. Our released split is *user-anonymous session-based*
  (test_pairs.json carries only an ordered POI-id history + target, NO user_id and
  NO absolute timestamps -- the same protocol every other baseline in the paper uses).
  We therefore re-implement the core dual-gate mechanism under our identical
  full-candidate leave-one-out protocol, realising the "time-interval" signal as
  *sequence-position recency* (the history is chronologically ordered, so position is a
  faithful recency proxy when absolute UTC time is not exposed). This is stated openly in
  the paper.

Protocol: identical to TIGER / Gwhere / ours -- full-candidate ranking, R@5/R@10/N@5/N@10.
"""

import os, sys, json, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_workspace():
    """向上搜索含 data/real_foursquare_nyc/processed/train_trajs.json 的目录。"""
    d = HERE
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "data", "real_foursquare_nyc", "processed", "train_trajs.json")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(HERE)


def _resolve_code_root():
    """向上搜索含 poi_bge_emb.npy 的目录（BGE 缓存在 code/ 根）。"""
    d = HERE
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "poi_bge_emb.npy")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(HERE)


CODE_ROOT = _resolve_code_root()
sys.path.insert(0, HERE)
sys.path.insert(0, CODE_ROOT)
sys.path.insert(0, os.path.join(CODE_ROOT, "src"))   # 确保 `from llm_stkg.evaluate import rank_metrics` 可用
from llm_stkg.evaluate import rank_metrics

MAX_LEN = 64
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_domain(city, data_root):
    ws = _resolve_workspace() if data_root is None else data_root
    if city in ("nyc", "tky"):
        proc = os.path.join(ws, "data", f"real_foursquare_{city}", "processed")
        tr_file = "train_trajs.json" if os.path.isfile(os.path.join(proc, "train_trajs.json")) else "train_checkins.json"
        te_file = "test_pairs.json" if os.path.isfile(os.path.join(proc, "test_pairs.json")) else "test_samples.json"
        raw_train = json.load(open(os.path.join(proc, tr_file), encoding="utf-8"))
        train_trajs = [t["pois"] for t in raw_train]
        test_pairs = json.load(open(os.path.join(proc, te_file), encoding="utf-8"))
        bge_cache = os.path.join(CODE_ROOT, "poi_bge_emb%s.npy" % ("" if city == "nyc" else "_tky"))
        bge = np.load(bge_cache) if os.path.isfile(bge_cache) else None
    else:
        proc = os.path.join(ws, "data", f"real_foursquare_{city}", "processed")
        raw_train = json.load(open(os.path.join(proc, "train_checkins.json"), encoding="utf-8"))
        train_trajs = [t["pois"] for t in raw_train]
        test_pairs = json.load(open(os.path.join(proc, "test_samples.json"), encoding="utf-8"))
        bge = None
    return train_trajs, test_pairs, bge


def revisit_ratio(pairs):
    rep = 0
    for p in pairs:
        h = set(int(x) for x in p["history"])
        if int(p["target"]) in h:
            rep += 1
    return rep / max(1, len(pairs))


def freq_rec_of(hist_ids):
    """给定历史 POI id 列表，返回每位置的归一化频次与近因 (position-based recency)。"""
    L = len(hist_ids)
    counts = {}
    for x in hist_ids:
        counts[x] = counts.get(x, 0) + 1
    maxc = max(counts.values()) if counts else 1
    freq = np.array([counts[x] / maxc for x in hist_ids], dtype=np.float32)
    rec = np.array([(i + 1) / L for i in range(L)], dtype=np.float32)   # 末端近因=1
    return freq, rec


class FreqTimeGRU(nn.Module):
    """Dual-gated recurrent cell: recency modulates the update (decay) gate; item-frequency
    down-weights the candidate state (popularity-bias mitigation). Final state = attention
    over history weighted by freq * recency."""

    def __init__(self, n_pois, d=64, dsub=16):
        super().__init__()
        self.d = d
        self.E = nn.Embedding(n_pois, d)
        self.fproj = nn.Linear(1, dsub)
        self.rproj = nn.Linear(1, dsub)
        inp = d + 2 * dsub
        self.Wz = nn.Linear(inp, d)
        self.Uz = nn.Linear(d, d)
        self.Wr = nn.Linear(inp, d)
        self.Ur = nn.Linear(d, d)
        self.Wh = nn.Linear(inp, d)
        self.Uh = nn.Linear(d, d)
        self.q = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.q, 0, 0.1)
        self.alpha = nn.Parameter(torch.tensor(0.4))   # 近因对遗忘门的调制强度
        self.beta = nn.Parameter(torch.tensor(0.4))    # 频次对候选状态的抑制强度

    def forward(self, hist, freq, rec, device):
        # hist: [B, L] long ; freq/rec: [B, L] float
        B, L = hist.shape
        x = self.E(hist)                                  # [B, L, d]
        fv = self.fproj(freq.unsqueeze(-1))               # [B, L, dsub]
        rv = self.rproj(rec.unsqueeze(-1))                # [B, L, dsub]
        x = torch.cat([x, fv, rv], dim=-1)                # [B, L, inp]
        h = torch.zeros(B, self.d, device=device)
        hs = []
        for t in range(L):
            xt = x[:, t, :]
            f = freq[:, t].unsqueeze(-1)                  # [B,1]
            r = rec[:, t].unsqueeze(-1)                   # [B,1]
            z = torch.sigmoid(self.Wz(xt) + self.Uz(h) + self.alpha * r)      # 近因↑ -> 保留更多
            rg = torch.sigmoid(self.Wr(xt) + self.Ur(h))
            htilde = torch.tanh(self.Wh(xt) + self.Uh(rg * h))
            htilde = htilde * (1.0 - self.beta * torch.clamp(f, 0, 1))        # 流行偏置抑制
            h = (1.0 - z) * h + z * htilde
            hs.append(h)
        H = torch.stack(hs, dim=1)                         # [B, L, d]
        # attention: (h_t . q) * sqrt(freq) * sqrt(rec)
        qe = torch.tanh(H @ self.q)                       # [B, L]
        aw = qe * torch.sqrt(freq + 1e-6) * torch.sqrt(rec + 1e-6)
        aw = F.softmax(aw, dim=1)                          # [B, L]
        hstar = (aw.unsqueeze(-1) * H).sum(dim=1)         # [B, d]
        return hstar


def build_prefix_samples(train_trajs, max_len=MAX_LEN):
    samples = []   # (hist_ids, next_poi)
    for tr in train_trajs:
        tr = [int(x) for x in tr if x is not None]
        if len(tr) < 2:
            continue
        for pos in range(1, len(tr)):
            hist = tr[:pos][-max_len:]
            samples.append((hist, tr[pos]))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["nyc", "tky", "steam", "gowalla", "ml1m", "steam200k", "amazon_beauty"])
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max_train", type=int, default=0, help="0=use all prefix samples; >0=randomly subsample this many for training (eval stays full-candidate)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    ev_dev = torch.device("cpu")   # eval 稳定（避免长循环 CUDA 上下文崩溃）
    train_trajs, test_pairs, bge = load_domain(args.city, args.data_root)
    num_pois = int(bge.shape[0]) if bge is not None else int(max(max(p) for p in train_trajs if p) + 1)
    print(f"[{args.city}] num_pois={num_pois} train_trajs={len(train_trajs)} test={len(test_pairs)} device={device}")

    model = FreqTimeGRU(num_pois, d=args.d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    samples = build_prefix_samples(train_trajs)
    if args.max_train and args.max_train < len(samples):
        rng0 = np.random.default_rng(SEED)
        idx_sub = rng0.choice(len(samples), size=args.max_train, replace=False)
        samples = [samples[i] for i in idx_sub]
        print(f"[{args.city}] prefix samples subsampled -> {len(samples)} (max_train={args.max_train})")
    else:
        print(f"[{args.city}] prefix samples={len(samples)}")
    # 预计算每前缀的频次/近因，避免每个 epoch 重复 O(L) 统计
    precomp = []
    for (h, _) in samples:
        f, r = freq_rec_of(h)
        precomp.append((np.asarray(f, np.float32), np.asarray(r, np.float32)))

    for ep in range(args.epochs):
        rng = np.random.default_rng(SEED + ep)
        perm = rng.permutation(len(samples))
        model.train()
        tot = 0.0
        nbat = 0
        for i in range(0, len(samples), args.batch):
            idx = perm[i:i + args.batch]
            hists, nxt = [], []
            for j in idx:
                hists.append(samples[j][0])
                nxt.append(samples[j][1])
            L = max(len(h) for h in hists)
            hist_t = torch.zeros(len(hists), L, dtype=torch.long)
            freq_t = torch.zeros(len(hists), L)
            rec_t = torch.zeros(len(hists), L)
            for bi, h in enumerate(hists):
                f, r = precomp[idx[bi]]
                hist_t[bi, :len(h)] = torch.tensor(h, dtype=torch.long)
                freq_t[bi, :len(h)] = torch.as_tensor(f)
                rec_t[bi, :len(h)] = torch.as_tensor(r)
            hist_t = hist_t.to(device)
            freq_t = freq_t.to(device)
            rec_t = rec_t.to(device)
            nxt_t = torch.tensor(nxt, dtype=torch.long, device=device)
            hstar = model(hist_t, freq_t, rec_t, device)
            S = hstar @ model.E.weight.t()                 # [B, N]
            loss = F.cross_entropy(S, nxt_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nbat += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[{args.city}] epoch {ep+1}/{args.epochs} loss={tot/max(1,nbat):.4f}")

    # ---- eval: full-candidate ranking ----
    model.to(ev_dev)
    model.eval()
    rv = revisit_ratio(test_pairs)
    mask_history = rv < 0.05
    print(f"[{args.city}] revisit_ratio={rv:.3f} mask_history={mask_history}")
    all_scores, all_tgt = [], []
    with torch.no_grad():
        for p in test_pairs:
            h = [int(x) for x in p["history"]][-MAX_LEN:]
            if len(h) == 0:
                h = [int(p["target"])]
            f, r = freq_rec_of(h)
            hist_t = torch.tensor([h], dtype=torch.long, device=ev_dev)
            freq_t = torch.as_tensor(f, device=ev_dev).unsqueeze(0)
            rec_t = torch.as_tensor(r, device=ev_dev).unsqueeze(0)
            hstar = model(hist_t, freq_t, rec_t, ev_dev)     # [1, d]
            S = (hstar @ model.E.weight.t()).squeeze(0).cpu().numpy()  # [N]
            if mask_history:
                for x in set(h):
                    S[x] = -1e9
            all_scores.append(S)
            all_tgt.append(int(p["target"]))
    scores_t = torch.tensor(np.stack(all_scores), dtype=torch.float32)
    tgt_t = torch.tensor(all_tgt, dtype=torch.long)
    m = rank_metrics(scores_t, tgt_t, k_list=(5, 10))
    out = {
        "model": "TiIfSRec (recipe-level, dual-gated time/frequency GRU)",
        "city": args.city,
        "protocol": "full-candidate leave-one-out",
        "R@5": m["Recall@5"], "R@10": m["Recall@10"],
        "N@5": m["NDCG@5"], "N@10": m["NDCG@10"],
        "revisit_ratio": round(rv, 4),
        "note": "Recipe-level re-implementation of TiIfSRec (Wang, uOttawa 2026 thesis). Time-interval gate realised as sequence-position recency (user-anonymous session split exposes no absolute timestamps). Learned POI embeddings; full-candidate ranking identical to all paper baselines.",
    }
    out_path = args.out or os.path.join(HERE, f"tiflsrec_{args.city}.json")
    json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"[{args.city}] TiIfSRec ->", {k: out[k] for k in ("R@5", "R@10", "N@5", "N@10")})
    print("wrote", out_path)


if __name__ == "__main__":
    main()
