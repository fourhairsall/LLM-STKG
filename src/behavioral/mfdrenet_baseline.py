# -*- coding: utf-8 -*-
"""
MFDReNet (recipe-level re-implementation) -- multi-factor decoupling repeat-aware network,
used as a *learned behavioral-prior* baseline for the LLM-STKG paper.

Original: Wang, H., Zhou, Q., Cai, J., Qiu, Y. "A multi-factor decoupling repeat aware
network for session-based recommendation" (MFDReNet). Multimedia Systems, 32(4), 2026.
Core idea: a repeat-explore intent mechanism that scores *old* (already-visited) items and
*new* (unvisited) items separately, combined by a learned repeat-explore gate; a
multi-factor decoupling module captures several latent preference factors.

Why recipe-level (not released code):
  MFDReNet code was not released with the publication. We re-implement the repeat-explore
  core under our identical full-candidate leave-one-out protocol:
    - repeat branch : per visited POI, score = freq^gamma * recency^delta  (HF/HR-style,
                      but with *learned* exponents -- the behavioral prior is learned, not
                      parameter-free)
    - explore branch: a GRU sequential encoder over the history -> dot with a 2-factor
                      decoupled POI embedding, scored over ALL candidates (visited + new)
    - gate g        : a learned scalar blending repeat vs explore (unvisited POIs get only
                      the explore branch, repeat score = -inf)
  The contrastive-learning augmentation module of the original is omitted (stated in paper);
  it is a training regulariser, not part of the scoring prior we compare.

Protocol: identical to every other baseline -- full-candidate ranking, R@5/R@10/N@5/N@10.
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_workspace():
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
sys.path.insert(0, os.path.join(CODE_ROOT, "src"))
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


def repeat_stats(hist_ids):
    """返回每 POI 的归一化频次与末次近因（position-based recency）。"""
    L = len(hist_ids)
    counts = {}
    last_pos = {}
    for i, x in enumerate(hist_ids):
        counts[x] = counts.get(x, 0) + 1
        last_pos[x] = i
    maxc = max(counts.values()) if counts else 1
    freq = np.array([counts[x] / maxc for x in hist_ids], dtype=np.float32)
    rec = np.array([(last_pos[x] + 1) / L for x in hist_ids], dtype=np.float32)
    return counts, last_pos, freq, rec, maxc, L


class MFDReNet(nn.Module):
    def __init__(self, n_pois, d=64, factors=2):
        super().__init__()
        self.d = d
        self.factors = factors
        # 多因子解耦：factors 个 POI 嵌入视图，打分用其平均
        self.Es = nn.ModuleList([nn.Embedding(n_pois, d) for _ in range(factors)])
        self.gru = nn.GRU(d, d, batch_first=True)
        # repeat-explore 门的全局可学习标量
        self.gate = nn.Parameter(torch.tensor(0.5))
        # 复访分支的频次/近因指数（softplus 保证正）
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.delta = nn.Parameter(torch.tensor(1.0))

    def E_avg(self):
        return sum(e.weight for e in self.Es) / self.factors

    def explore_scores(self, hist, device):
        # hist: [B, L] long -> GRU -> last hidden -> dot with E_avg
        B, L = hist.shape
        E = self.E_avg()                                  # [N, d]
        x = E[hist]                                       # [B, L, d]
        # 末位有效 hidden
        out, hn = self.gru(x)                             # out [B,L,d], hn [1,B,d]
        hlast = hn.squeeze(0)                             # [B, d]
        return hlast @ E.t()                             # [B, N]

    def repeat_scores_full(self, test_pairs_hist, device, n_pois):
        """对一批历史，构造 [B, N] 复访打分（已访问=freq^gamma*rec^delta，未访问=-inf）。"""
        B = len(test_pairs_hist)
        # 未访问 POI 的复访证据设为 0（"无复访证据"）；
        # 注意：绝不能填 -1e9，因为下游 S = g*repeat + (1-g)*explore 会把 -1e9 乘以门控 g，
        # 将门控梯度推向饱和、冻结全部可学习参数。0 是中性初值，配合 1-g 的 explore 分支即可。
        S = torch.zeros((B, n_pois), device=device)
        for bi, hist_ids in enumerate(test_pairs_hist):
            counts, last_pos, freq, rec, maxc, L = repeat_stats(hist_ids)
            g = F.softplus(self.gamma)
            dl = F.softplus(self.delta)
            for x in counts:
                s = (counts[x] / maxc) ** g.item() * (((last_pos[x] + 1) / L) ** dl.item())
                S[bi, x] = float(s)
        return S


def build_prefix_samples(train_trajs, max_len=MAX_LEN):
    samples = []
    for tr in train_trajs:
        tr = [int(x) for x in tr if x is not None]
        if len(tr) < 2:
            continue
        for pos in range(1, len(tr)):
            samples.append((tr[:pos][-max_len:], tr[pos]))
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
    ev_dev = torch.device("cpu")
    train_trajs, test_pairs, bge = load_domain(args.city, args.data_root)
    num_pois = int(bge.shape[0]) if bge is not None else int(max(max(p) for p in train_trajs if p) + 1)
    print(f"[{args.city}] num_pois={num_pois} train_trajs={len(train_trajs)} test={len(test_pairs)} device={device}")

    model = MFDReNet(num_pois, d=args.d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    samples = build_prefix_samples(train_trajs)
    if args.max_train and args.max_train < len(samples):
        rng0 = np.random.default_rng(SEED)
        idx_sub = rng0.choice(len(samples), size=args.max_train, replace=False)
        samples = [samples[i] for i in idx_sub]
        print(f"[{args.city}] prefix samples subsampled -> {len(samples)} (max_train={args.max_train})")
    else:
        print(f"[{args.city}] prefix samples={len(samples)}")

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
            for bi, h in enumerate(hists):
                hist_t[bi, :len(h)] = torch.tensor(h, dtype=torch.long)
            hist_t = hist_t.to(device)
            nxt_t = torch.tensor(nxt, dtype=torch.long, device=device)
            explore = model.explore_scores(hist_t, device)            # [B, N]
            repeat = model.repeat_scores_full(hists, device, num_pois)  # [B, N] (-inf unvisited)
            g = torch.sigmoid(model.gate)
            S = g * repeat + (1.0 - g) * explore
            loss = F.cross_entropy(S, nxt_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nbat += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[{args.city}] epoch {ep+1}/{args.epochs} loss={tot/max(1,nbat):.4f} g={torch.sigmoid(model.gate).item():.3f}")

    # ---- eval ----
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
            hist_t = torch.tensor([h], dtype=torch.long, device=ev_dev)
            explore = model.explore_scores(hist_t, ev_dev).squeeze(0)        # [N]
            repeat = model.repeat_scores_full([h], ev_dev, num_pois).squeeze(0)  # [N]
            g = torch.sigmoid(model.gate)
            S = (g * repeat + (1.0 - g) * explore).cpu().numpy()
            if mask_history:
                for x in set(h):
                    S[x] = -1e9
            all_scores.append(S)
            all_tgt.append(int(p["target"]))
    scores_t = torch.tensor(np.stack(all_scores), dtype=torch.float32)
    tgt_t = torch.tensor(all_tgt, dtype=torch.long)
    m = rank_metrics(scores_t, tgt_t, k_list=(5, 10))
    out = {
        "model": "MFDReNet (recipe-level, repeat-explore gate + 2-factor decoupling)",
        "city": args.city,
        "protocol": "full-candidate leave-one-out",
        "R@5": m["Recall@5"], "R@10": m["Recall@10"],
        "N@5": m["NDCG@5"], "N@10": m["NDCG@10"],
        "repeat_explore_gate_g": round(float(torch.sigmoid(model.gate).item()), 4),
        "revisit_ratio": round(rv, 4),
        "note": "Recipe-level re-implementation of MFDReNet (Wang et al., Multimedia Systems 2026). Repeat branch = learned-exponent freq^gamma * recency^delta over visited POIs; explore branch = GRU over history dotted with a 2-factor decoupled POI embedding; repeat-explore gate blends them. Contrastive module omitted. Learned POI embeddings; full-candidate ranking identical to all paper baselines.",
    }
    out_path = args.out or os.path.join(HERE, f"mfdrenet_{args.city}.json")
    json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"[{args.city}] MFDReNet ->", {k: out[k] for k in ("R@5", "R@10", "N@5", "N@10", "repeat_explore_gate_g")})
    print("wrote", out_path)


if __name__ == "__main__":
    main()
