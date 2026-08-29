"""
Gwhere-style generative-retrieval baseline -- a RECIPE-LEVEL reproduction of
"Guess Where You Go: Generative Next Point-of-Interest Recommendation in Amap"
(Zhai et al., RecSys '26, arXiv:2607.26073) -- for head-to-head comparison
against LLM-STKG under the IDENTICAL full-candidate leave-one-out protocol.

Why this method (and what is faithfully reproduced)
---------------------------------------------------
Gwhere's central novelty over vanilla TIGER-style generative retrieval is its
*contrastive residual-quantization Semantic-ID tokenizer*. Instead of quantizing
raw text embeddings (TIGER), Gwhere fuses four modalities -- text, visual,
spatial, collaborative -- into a single discriminative item representation via
attention fusion, then trains the SID codebooks with an NT-Xent contrastive loss
on co-visitation (NOT a reconstruction loss). The fused, contrastively-trained
representation is what makes the resulting SIDs well-separated and predictable.

What we reproduce here (the part that is both novel to Gwhere and reproducible
on a single GPU):
  1. Multi-modal attention fusion of (a) BGE text semantics, (b) spatial coords
     via a Haversine-style spatial projector, (c) a learnable collaborative
     embedding aligned by co-visitation. [visual modality omitted: Foursquare
     POIs carry no imagery -- the other three modalities are the comparable core]
  2. NT-Xent contrastive pretraining of the POI encoder on co-visitation pairs
     (tau = 0.1, exactly as Gwhere).
  3. Residual quantization (K-means RQ-VAE recipe, identical capacity to TIGER)
     of the fused embedding -> Semantic-ID tuple.
  4. A small decoder-only Transformer trained autoregressively to generate the
     target POI's Semantic-ID from the history (beam search decode -> POI).

Honest scope limits (disclosed in the paper)
--------------------------------------------
  * Gwhere's published backbone is Qwen2.5 (0.5B-7B) adapted via CPT + SFT + an
    EAKTO reinforcement-learning objective, trained on 200xH20 GPUs; that code
    is NOT released (repo marks "LLM training: coming soon") and is non-runnable
    here. We therefore replace the LLM backbone with the SAME small decoder-only
    Transformer used for the TIGER baseline, so the two generative baselines are
    directly comparable to each other and to ours.
  * EAKTO (the RL preference-alignment stage) is omitted for the same reason.
  * The evaluation protocol is OUR paper's full-candidate leave-one-out R@K /
    NDCG@K, NOT Gwhere's reported Acc@1 under an 80/10/10 split. This is the
    correct head-to-head setting; we cite Gwhere's own Acc@1 numbers separately.
This is a defensible recipe-level reproduction: it isolates Gwhere's SID-tokenizer
contribution (the part that distinguishes it from TIGER) while keeping the rest of
the pipeline at the same comparable scale.
"""
import os, sys, json, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Force single-threaded torch/BLAS to avoid the OpenMP/OpenBLAS segfault that
# otherwise appears under sustained loops of many small CUDA launches.
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
def _resolve_workspace():
    """Walk up from this file until we find the data root that holds the
    Foursquare-NYC processed split (identified by train_trajs.json)."""
    d = HERE
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "data", "real_foursquare_nyc", "processed", "train_trajs.json")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(HERE)
WORKSPACE = _resolve_workspace()
def _resolve_code_root():
    """Walk up from this file until we find the dir holding the BGE cache."""
    d = HERE
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "poi_bge_emb.npy")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(HERE)
CODE_ROOT = _resolve_code_root()  # .../code  (BGE caches live here)
SRC = os.path.join(CODE_ROOT, "src")  # .../code/src  (llm_stkg package lives here)
sys.path.insert(0, HERE)
sys.path.insert(0, SRC)
sys.path.insert(0, CODE_ROOT)
from llm_stkg.evaluate import rank_metrics

SEED = 42
MAX_LEN = 64
L_CODES = 3          # Semantic-ID depth (same as TIGER for fair comparison)
C_BOOK = 256         # codebook size per level (same as TIGER)
D_FUSE = 64          # fused multimodal embedding dim (== TIGER RQ-VAE latent dim)
D_MODEL = 128        # Transformer dim (same as TIGER)
N_LAYERS = 3
N_HEADS = 4
PAD_ID = 0
SEP_ID = 1
def code_token(level, code):
    return 2 + level * C_BOOK + code
def max_code_token():
    return 2 + L_CODES * C_BOOK

torch.manual_seed(SEED)
np.random.seed(SEED)


# ----------------------------- Semantic-ID (residual quantization) -----------------------------
def _kmeans_assign(cb, residuals):
    cb_n = np.einsum("kd,kd->k", cb, cb)
    r_n = np.einsum("nd,nd->n", residuals, residuals)
    sim = residuals @ cb.T
    d2 = r_n[:, None] + cb_n[None, :] - 2.0 * sim
    return np.argmin(d2, axis=1).astype(np.int64)


def kmeans_residual_quantize(X, n_levels=L_CODES, codebook_size=C_BOOK, n_iter=50, seed=SEED):
    rng = np.random.default_rng(seed)
    residuals = np.ascontiguousarray(X, dtype=np.float32).copy()
    N = residuals.shape[0]
    all_codes = []
    for lv in range(n_levels):
        perm = rng.permutation(N)[:codebook_size]
        cb = residuals[perm].astype(np.float32).copy()
        for _ in range(n_iter):
            idx = _kmeans_assign(cb, residuals)
            new_cb = cb.copy()
            for k in range(codebook_size):
                m = idx == k
                if m.any():
                    new_cb[k] = residuals[m].mean(0)
            counts = np.bincount(idx, minlength=codebook_size)
            dead = np.where(counts == 0)[0]
            if dead.size:
                for k in dead:
                    new_cb[k] = residuals[rng.integers(0, N)]
            cb = new_cb.astype(np.float32)
        idx = _kmeans_assign(cb, residuals)
        all_codes.append(idx.astype(np.int64))
        residuals = residuals - cb[idx]
    return np.stack(all_codes, 1)


# ----------------------------- Gwhere multi-modal contrastive encoder -----------------------------
class GwhereEncoder(nn.Module):
    """Attention-fused text + spatial + collaborative POI encoder, contrastively
    pretrained on co-visitation (NT-Xent). Mirrors Gwhere's tokenization stage."""
    def __init__(self, num_pois, bge_dim, d=D_FUSE):
        super().__init__()
        self.text_proj = nn.Linear(bge_dim, d)
        self.spatial_proj = nn.Sequential(nn.Linear(2, d), nn.ReLU(), nn.Linear(d, d))
        self.cf_emb = nn.Embedding(num_pois, d)          # collaborative signal (learned from co-visitation)
        self.q = nn.Parameter(torch.zeros(d))            # attention query over modalities
        self.g = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))  # projector for NT-Xent

    def fuse(self, bge, spat, idx):
        zt = self.text_proj(bge)            # [N, d]
        zs = self.spatial_proj(spat)        # [N, d]
        zc = self.cf_emb(idx)               # [N, d]
        Z = torch.stack([zt, zs, zc], 1)    # [N, 3, d]
        a = torch.softmax((Z * self.q).sum(-1) / math.sqrt(Z.size(-1)), 1)  # [N, 3]
        z = (a.unsqueeze(-1) * Z).sum(1)    # [N, d]
        return z, a

    def project(self, z):
        return F.normalize(self.g(z), dim=-1)


def build_covisit_pairs(train_trajs, num_pois, topk=8, min_co=2):
    """Co-visitation graph -> (anchor, positive) pairs (top-k co-visited POIs)."""
    from collections import defaultdict
    cnt = defaultdict(lambda: defaultdict(int))
    for t in train_trajs:
        u = sorted(set(int(x) for x in t))
        for i in range(len(u)):
            for j in range(len(u)):
                if i != j:
                    cnt[u[i]][u[j]] += 1
    pairs = []
    for p, nbrs in cnt.items():
        top = sorted(nbrs.items(), key=lambda kv: -kv[1])[:topk]
        for q, c in top:
            if c >= min_co and p < num_pois and q < num_pois:
                pairs.append((p, q))
    return pairs


def nt_xent(h_a, h_p, tau=0.1):
    """Symmetric NT-Xent over explicit (anchor, positive) pairs (in-batch negs)."""
    B = h_a.size(0)
    h = torch.cat([h_a, h_p], 0)            # [2B, d] normalized
    sim = h @ h.T / tau                      # [2B, 2B]
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(h.device)
    # mask self-similarity
    eye = torch.eye(2 * B, device=h.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2.0
    return loss


# ----------------------------- Transformer (identical to TIGER) -----------------------------
class GenRet(nn.Module):
    def __init__(self, vocab, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, n_layers)
        self.head = nn.Linear(d_model, vocab)
        self.vocab = vocab

    def forward(self, tgt, tgt_mask=None):
        h = self.embed(tgt)
        h = self.dec(h, h, tgt_mask=tgt_mask)
        return self.head(h)


def causal_mask(sz, device):
    return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()


# ----------------------------- data -----------------------------
def load_domain(city, data_root):
    if city in ("nyc", "tky"):
        sub = "real_foursquare_%s" % city
        processed_dir = os.path.join(WORKSPACE, "data", sub, "processed")
        raw_train = json.load(open(os.path.join(processed_dir, "train_trajs.json"), encoding="utf-8"))
        train_trajs = [t["pois"] for t in raw_train]
        test_pairs = json.load(open(os.path.join(processed_dir, "test_pairs.json"), encoding="utf-8"))
        bge_cache = os.path.join(CODE_ROOT, "poi_bge_emb%s.npy" % ("" if city == "nyc" else "_tky"))
        meta_path = os.path.join(processed_dir, "poi_meta.json")
    else:
        _cross_map = {"steam": "steam", "gowalla": "gowalla", "ml1m": "ml-1m"}
        sub = data_root or _cross_map.get(city, city)
        processed_dir = os.path.join(WORKSPACE, "data", sub, "processed")
        raw_train = json.load(open(os.path.join(processed_dir, "train_checkins.json"), encoding="utf-8"))
        train_trajs = [t["pois"] for t in raw_train]
        test_pairs = json.load(open(os.path.join(processed_dir, "test_samples.json"), encoding="utf-8"))
        bge_cache = None
        meta_path = None
    bge = np.load(bge_cache).astype(np.float32) if (bge_cache and os.path.exists(bge_cache)) else None
    meta = json.load(open(meta_path, encoding="utf-8")) if meta_path and os.path.exists(meta_path) else None
    return train_trajs, test_pairs, bge, meta


def revisit_ratio(pairs):
    n = r = 0
    for p in pairs:
        h = set(int(x) for x in p["history"])
        if int(p["target"]) in h:
            r += 1
        n += 1
    return r / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["nyc", "tky", "steam", "gowalla", "ml1m"], required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--epochs_contrastive", type=int, default=30, help="NT-Xent contrastive pretraining epochs")
    ap.add_argument("--topk_pos", type=int, default=8, help="top-k co-visited POIs kept as positives")
    ap.add_argument("--epochs_rq", type=int, default=50, help="K-means residual-quantization iterations per level")
    ap.add_argument("--epochs_tr", type=int, default=30)
    ap.add_argument("--codebook", type=int, default=256)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    global C_BOOK, L_CODES
    C_BOOK = args.codebook
    L_CODES = args.levels
    if args.out is None:
        args.out = os.path.join(HERE, f"gwhere_{args.city}.json")
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_trajs, test_pairs, bge, meta = load_domain(args.city, args.data_root)
    num_pois = int(bge.shape[0])
    print(f"[{args.city}] num_pois={num_pois} train_trajs={len(train_trajs)} test={len(test_pairs)} device={device}")

    # ---- spatial features (lat/lng), normalized per dataset ----
    lat = np.zeros(num_pois, np.float32); lng = np.zeros(num_pois, np.float32)
    if meta is not None:
        for pid, info in meta.items():
            i = int(pid)
            if 0 <= i < num_pois:
                lat[i] = float(info.get("lat", 0.0)); lng[i] = float(info.get("lng", 0.0))
    spat = np.stack([lat, lng], 1)
    spat = (spat - spat.mean(0, keepdims=True)) / (spat.std(0, keepdims=True) + 1e-6)
    spat = torch.tensor(spat, dtype=torch.float32)

    bge_t = torch.tensor(bge, dtype=torch.float32)
    idx_t = torch.arange(num_pois)

    # ---- (1) contrastive residual-quantization SID: pretrain fused encoder on co-visitation ----
    pairs = build_covisit_pairs(train_trajs, num_pois, topk=args.topk_pos)
    print(f"[{args.city}] co-visitation positive pairs: {len(pairs)}")
    enc = GwhereEncoder(num_pois, bge_dim=bge.shape[1], d=D_FUSE).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    rng = np.random.default_rng(SEED)
    pidx = np.array(pairs, dtype=np.int64)
    for ep in range(args.epochs_contrastive):
        perm = rng.permutation(len(pidx))
        tot = 0.0; nb = 0
        for i in range(0, len(pidx), args.batch):
            b = pidx[perm[i:i + args.batch]]
            a, p = b[:, 0], b[:, 1]
            za, _ = enc.fuse(bge_t[a].to(device), spat[a].to(device), torch.tensor(a, device=device))
            zp, _ = enc.fuse(bge_t[p].to(device), spat[p].to(device), torch.tensor(p, device=device))
            ha = enc.project(za); hp = enc.project(zp)
            loss = nt_xent(ha, hp, tau=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"[{args.city}] Gwhere contrastive epoch {ep+1} loss={tot/nb:.4f}", flush=True)

    # fused representation for all POIs (detached) -> residual quantize
    enc.eval()
    with torch.no_grad():
        z_all, _ = enc.fuse(bge_t.to(device), spat.to(device), idx_t.to(device))
        z_np = z_all.cpu().numpy().astype(np.float32)
    sem_ids = kmeans_residual_quantize(z_np, n_levels=L_CODES, codebook_size=C_BOOK,
                                       n_iter=args.epochs_rq, seed=SEED)
    uniq = len(set(tuple(sem_ids[i]) for i in range(num_pois)))
    print(f"[{args.city}] Semantic-ID uniqueness: {uniq} / {num_pois} (codebook {C_BOOK}^{L_CODES})")

    # ---- (2) build training sequences of code tokens (identical to TIGER) ----
    def to_tokens(traj):
        toks = []
        for poi in traj[-MAX_LEN:]:
            for lv in range(L_CODES):
                toks.append(code_token(lv, int(sem_ids[poi, lv])))
        return toks
    seqs = []
    for t in train_trajs:
        if len(t) < 2:
            continue
        hist = to_tokens(t[:-1]); tgt = to_tokens([t[-1]])
        seqs.append(hist + [SEP_ID] + tgt)
    max_seq = MAX_LEN * L_CODES * 2 + 2
    seqs = [s[-max_seq:] for s in seqs if len(s) <= max_seq]

    vocab = max_code_token() + 2
    model = GenRet(vocab).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    cmask = causal_mask(max_seq, device)
    for ep in range(args.epochs_tr):
        perm = torch.randperm(len(seqs))
        tot = 0.0; nb = 0
        for i in range(0, len(seqs), args.batch):
            batch_seqs = [seqs[j] for j in perm[i:i + args.batch]]
            L = max(len(s) for s in batch_seqs)
            inp = torch.full((len(batch_seqs), L), PAD_ID, dtype=torch.long, device=device)
            for bi, s in enumerate(batch_seqs):
                inp[bi, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            tgt_in = inp[:, :-1]; tgt_out = inp[:, 1:]
            m = cmask[:tgt_in.size(1), :tgt_in.size(1)]
            logits = model(tgt_in, tgt_mask=m)
            loss = F.cross_entropy(logits.reshape(-1, vocab), tgt_out.reshape(-1), ignore_index=PAD_ID)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"[{args.city}] Gwhere decoder epoch {ep+1} loss={tot/nb:.4f}", flush=True)

    # ---- (3) eval: beam search target Semantic ID -> POI (on CPU, stable) ----
    model.eval(); model = model.cpu(); ev_dev = torch.device("cpu")
    cmask_ev = causal_mask(max_seq, ev_dev)
    id2poi = {tuple(int(sem_ids[poi, lv]) for lv in range(L_CODES)): poi for poi in range(num_pois)}
    all_scores, all_tgt = [], []
    rv = revisit_ratio(test_pairs)
    mask_history = rv < 0.05
    done = 0
    with torch.no_grad():
        for p in test_pairs:
            h_raw = [int(x) for x in p["history"]][-MAX_LEN:]
            if not h_raw:
                continue
            hist = []
            for poi in h_raw:
                for lv in range(L_CODES):
                    hist.append(code_token(lv, int(sem_ids[poi, lv])))
            beams = [(0.0, [])]
            for lv in range(L_CODES):
                valid = torch.arange(2 + lv * C_BOOK, 2 + (lv + 1) * C_BOOK, device=ev_dev)
                cands = []
                for logp, prefix in beams:
                    seq = hist + [SEP_ID] + [code_token(k, prefix[k]) for k in range(lv)]
                    x = torch.tensor(seq, dtype=torch.long, device=ev_dev).unsqueeze(0)
                    m = cmask_ev[:x.size(1), :x.size(1)]
                    logits = model(x, tgt_mask=m)[0, -1, valid]
                    top = torch.topk(logits, args.beam)
                    for val, j in zip(top.values, top.indices):
                        code = int(valid[j]) - (2 + lv * C_BOOK)
                        cands.append((logp + float(val), prefix + [code]))
                cands.sort(key=lambda t: -t[0])
                beams = cands[:args.beam]
            score_vec = torch.full((num_pois,), -1e9)
            for logp, prefix in beams:
                poi = id2poi.get(tuple(prefix))
                if poi is not None:
                    score_vec[poi] = -logp
            if mask_history:
                for x in h_raw:
                    if x < num_pois:
                        score_vec[x] = -1e9
            all_scores.append(score_vec.unsqueeze(0))
            all_tgt.append(int(p["target"]))
            done += 1
            if done % 300 == 0:
                print(f"[{args.city}] eval {done}/{len(test_pairs)}", flush=True)
    scores = torch.cat(all_scores, 0)
    targets = torch.tensor(all_tgt)
    metrics = rank_metrics(scores, targets, k_list=(1, 5, 10))
    print(f"[{args.city}] Gwhere metrics:", metrics)
    json.dump({"model": "Gwhere-style (contrastive residual-quantization Semantic-ID + generative retrieval, "
                       "recipe-level reproduction of Gwhere, RecSys '26, arXiv:2607.26073)",
               "city": args.city, "k_list": [1, 5, 10], "metrics": metrics,
               "n_test": int(len(targets)),
               "semantic_id_uniqueness": uniq,
               "note": "Recipe-level Gwhere reproduction (contrastive RQ SID; LLM backbone=small decoder-only "
                       "Transformer, EAKTO omitted, no visual modality) under the paper's full-candidate LOO "
                       "protocol, to benchmark the latest 2026 generative POI SOTA and close reviewer critique #3."},
              open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
