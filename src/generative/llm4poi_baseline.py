"""
LLM4POI-style generative baseline for head-to-head comparison.

We fine-tune Qwen2.5-3B-Instruct with LoRA as a *POI sequential language model*:
each POI is a special token <POI_i>; a trajectory is a sequence of POI tokens and
the model is trained with next-POI-token prediction. At evaluation we take the
logits over ALL POI tokens at the last position and rank the true target among all
POIs -- this is exactly the full-candidate protocol used by LLM4POI and identical
to the protocol of our LLM-STKG (we reuse llm_stkg.evaluate.rank_metrics).

Key honesty notes:
  * POI token embeddings are INITIALISED from BGE semantic vectors (hidden = [bge|0]);
    this injects the strongest semantic prior available, mirroring LLM4POI's
    geospatial-embedding initialisation, so any under-performance is not due to a
    weak init strawman.
  * We use LoRA (a standard PEFT), labelled LLM4POI-style, not a faithful P-tuning-v2
    reproduction.
"""
import os, sys, json, argparse, math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model

HERE = os.path.dirname(os.path.abspath(__file__))
def _resolve_workspace():
    """Walk up from this file until we find the data root that holds the
    Foursquare-NYC processed split (identified by train_trajs.json, not just
    the dir name, to skip empty placeholder dirs). Robust to the repo layout
    (scripts live under src/generative/ after reorganisation)."""
    d = HERE
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "data", "real_foursquare_nyc", "processed", "train_trajs.json")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(HERE)  # fallback: code/
WORKSPACE = _resolve_workspace()
def _resolve_code_root():
    """Walk up from this file until we find the dir holding the BGE cache
    (poi_bge_emb*.npy). Robust to repo layout after reorganisation."""
    d = HERE
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "poi_bge_emb.npy")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(HERE)  # fallback
CODE_ROOT = _resolve_code_root()  # .../code  (BGE caches live here)
sys.path.insert(0, HERE)
sys.path.insert(0, CODE_ROOT)
from llm_stkg.evaluate import rank_metrics

MAX_LEN = 64
SEED = 42

class TrajDataset(Dataset):
    def __init__(self, trajs, poi_tid, max_len=MAX_LEN, aug_max=0):
        """POI sequence LM samples.

        aug_max > 0: sliding-window augmentation -- each trajectory yields up to
        aug_max sub-sequences (covering positions 2..L), fixing the severe
        under-training that one-sample-per-trajectory causes on small datasets
        (Steam/Gowalla have only ~2.5k trajectories).
        """
        self.samples = []
        for t in trajs:
            t = [poi_tid[int(x)] for x in t]
            L = len(t)
            if L < 2:
                continue
            if aug_max and L > 2:
                if L <= aug_max:
                    idxs = list(range(2, L + 1))
                else:
                    step = (L - 1) / aug_max
                    idxs = sorted({min(L, 2 + int(round(k * step))) for k in range(aug_max)})
                for i in idxs:
                    self.samples.append(t[max(0, i - max_len):i])
            else:
                self.samples.append(t[-max_len:])
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        ids = self.samples[i]
        labels = ids[1:] + [-100]
        return {"input_ids": ids, "labels": labels}

def collate(batch):
    maxl = max(len(b["input_ids"]) for b in batch)
    inp, lab, msk = [], [], []
    for b in batch:
        L = len(b["input_ids"])
        inp.append(b["input_ids"] + [PAD] * (maxl - L))
        lab.append(b["labels"] + [-100] * (maxl - L))
        msk.append([1] * L + [0] * (maxl - L))
    return {"input_ids": torch.tensor(inp), "labels": torch.tensor(lab), "attention_mask": torch.tensor(msk)}

class TestDataset(Dataset):
    def __init__(self, test_pairs, poi_tid):
        self.hist, self.tgt, self.hist_raw = [], [], []
        for p in test_pairs:
            h_raw = [int(x) for x in p["history"]][-MAX_LEN:]
            if not h_raw:
                continue
            self.hist.append([poi_tid[x] for x in h_raw])
            self.tgt.append(int(p["target"]))
            self.hist_raw.append(h_raw)
    def __len__(self):
        return len(self.hist)
    def __getitem__(self, i):
        return {"input_ids": self.hist[i], "target": self.tgt[i], "hist_raw": self.hist_raw[i]}

def test_collate(batch):
    maxl = max(len(b["input_ids"]) for b in batch)
    inp, msk, tgt, hraw = [], [], [], []
    for b in batch:
        L = len(b["input_ids"])
        inp.append(b["input_ids"] + [PAD] * (maxl - L))
        msk.append([1] * L + [0] * (maxl - L))
        tgt.append(b["target"])
        hraw.append(b["hist_raw"])
    return {"input_ids": torch.tensor(inp), "attention_mask": torch.tensor(msk),
            "target": torch.tensor(tgt), "hist_raw": hraw}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["nyc", "tky", "steam", "gowalla", "ml1m",
                                        "steam200k", "amazon_beauty"], required=True)
    ap.add_argument("--data_root", default=None,
                    help="override processed dir (used for steam/gowalla/ml1m cross-domain runs)")
    ap.add_argument("--model_dir", default=os.path.join(HERE, "models", "Qwen2.5-3B-Instruct"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=100000)
    ap.add_argument("--aug_max", type=int, default=0,
                    help="sliding-window augmentation cap per trajectory (0=off, one sample per traj)")
    ap.add_argument("--no_bge", action="store_true",
                    help="disable BGE semantic seeding even when a cache exists (causal ablation)")
    ap.add_argument("--no_grad_ckpt", action="store_true",
                    help="disable gradient checkpointing (3B + short seqs: ~1.75x faster, no OOM risk)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(HERE, f"llm4poi_{args.city}.json")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    global PAD

    city = args.city
    if args.data_root:
        processed_dir = args.data_root
    else:
        # 跨域数据默认落在 data/<name>/processed（与 head_to_head 的 generic_loaders 输出一致）
        _cross_map = {"steam": "steam", "gowalla": "gowalla", "ml1m": "ml-1m",
                      "steam200k": "steam200k", "amazon_beauty": "amazon_beauty"}
        _sub = _cross_map.get(city, "real_foursquare_%s" % city)
        processed_dir = os.path.join(WORKSPACE, "data", _sub, "processed")
    # ---- data loading (per-domain formats) ----
    bge_cache = None
    pois = None
    if city in ("nyc", "tky"):
        raw_train = json.load(open(os.path.join(processed_dir, "train_trajs.json"), encoding="utf-8"))
        train_trajs = [t["pois"] for t in raw_train]  # each: list of remapped POI ids
        test_pairs = json.load(open(os.path.join(processed_dir, "test_pairs.json"), encoding="utf-8"))
        bge_cache = os.path.join(CODE_ROOT, "poi_bge_emb%s.npy" % ("" if city == "nyc" else "_tky"))
        num_pois = int(np.load(bge_cache).shape[0])  # BGE cache aligned with POI indices
    else:
        # cross-domain processed (generic_loaders output)
        raw_train = json.load(open(os.path.join(processed_dir, "train_checkins.json"), encoding="utf-8"))
        train_trajs = [t["pois"] for t in raw_train]
        test_pairs = json.load(open(os.path.join(processed_dir, "test_samples.json"), encoding="utf-8"))
        pois = json.load(open(os.path.join(processed_dir, "pois.json"), encoding="utf-8"))
        num_pois = int(len(pois))
        # BGE 语义种子缓存：文本可得域才有（ours 跨域统一 w/o LLM-text，但 LLM4POI-style
        # 基线需要文本做语义种子，以公平对照"文本可得 → LLM4POI-style 生效"）
        _bge_map = {"ml1m": "poi_bge_emb_ml1m.npy",
                    "steam200k": "poi_bge_emb_steam200k.npy",
                    "amazon_beauty": "poi_bge_emb_amazonbeauty.npy"}
        bge_cache = os.path.join(CODE_ROOT, _bge_map[city]) if city in _bge_map else None
    # mask-history: 与 head_to_head 自动协议完全一致 —— 测试端重访率 < 0.05 则开启
    # （SASRec 一系标准协议：在历史里 ⇒ 不是答案，屏蔽已交互物品）。无重复消费域
    # （steam/ml1m/steam200k/amazon_beauty，revisit≈0）开启；重访主导域（foursquare）关闭。
    def _revisit_ratio(pairs):
        n = 0; r = 0
        for _p in pairs:
            _h = set(int(x) for x in _p["history"])
            if int(_p["target"]) in _h:
                r += 1
            n += 1
        return (r / n) if n else 0.0
    _rv_test = _revisit_ratio(test_pairs)
    mask_history = _rv_test < 0.05
    print(f"[{city}] num_pois={num_pois} train_trajs={len(train_trajs)} "
          f"test={len(test_pairs)} revisit_ratio_test={_rv_test:.4f} "
          f"mask_history={'ON' if mask_history else 'OFF'}")
    if pois is not None and len(pois) > 0:
        print(f"[{city}] poi text sample: {str(pois[0].get('text', ''))[:70]!r}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    PAD = tokenizer.pad_token_id

    # add POI special tokens
    poi_tokens = [f"<POI_{i}>" for i in range(num_pois)]
    tokenizer.add_tokens(poi_tokens)
    base = tokenizer.convert_tokens_to_ids(poi_tokens[0])

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda()
    model.resize_token_embeddings(len(tokenizer))
    poi_tid = list(range(base, base + num_pois))
    poi_tid_arr = np.array(poi_tid)

    # semantic init of POI token embeddings from BGE (hidden = [bge | 0]); skip if no cache
    hidden = model.config.hidden_size
    bge = None
    if bge_cache and os.path.exists(bge_cache) and not args.no_bge:
        bge = torch.as_tensor(np.load(bge_cache), dtype=torch.float32)  # [N,768]
    with torch.no_grad():
        w = model.get_input_embeddings().weight
        if bge is not None:
            init = torch.zeros(num_pois, hidden)
            init[:, :bge.shape[1]] = bge
            w[base:base + num_pois] = init.to(w.dtype)
        else:
            # ID-only control: shrink the random init so early logits stay finite
            # (bf16 overflow guard; BGE-seeded rows are L2-normalised already)
            w[base:base + num_pois] = w[base:base + num_pois].to(w.dtype) * 0.1
    model.get_input_embeddings().weight.requires_grad_(True)  # learn POI reps

    # LoRA
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"],
                          task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, lora_cfg)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # ---- train ----
    tr_ds = TrajDataset(train_trajs, poi_tid, aug_max=args.aug_max)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    print(f"[{city}] train samples after aug={len(tr_ds)}")
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    # gstep counts OPTIMIZER steps (not micro-batches)
    total_opt = args.epochs * (len(tr_dl) // args.grad_accum + 1)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=max(1, total_opt // 20), num_training_steps=total_opt)
    model.train()
    mb = 0          # micro-batch counter (drives grad-accum)
    gstep = 0       # optimizer-step counter
    for ep in range(args.epochs):
        for batch in tr_dl:
            batch = {k: v.cuda() for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            if not torch.isfinite(loss):
                # skip pathological batch (rare; keeps training stable)
                print(f"[{args.city}] WARN non-finite loss at mb={mb}, skipping", flush=True)
                mb += 1
                continue
            loss.backward()
            mb += 1
            if mb % 10 == 0:
                print(f"[{args.city}] mb={mb} gstep={gstep} loss={loss.item():.4f}", flush=True)
            if mb % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                opt.step(); sched.step(); opt.zero_grad()
                gstep += 1
            if mb >= args.max_steps:
                break
        if mb % args.grad_accum != 0:   # flush residual gradient at epoch end
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            opt.step(); sched.step(); opt.zero_grad()
            gstep += 1
        print(f"[{args.city}] epoch {ep+1} done, opt_steps={gstep}, loss={loss.item():.4f}")
        if mb >= args.max_steps:
            break

    # ---- eval (full-candidate, same protocol as LLM-STKG) ----
    model.eval()
    te_ds = TestDataset(test_pairs, poi_tid)
    te_dl = DataLoader(te_ds, batch_size=32, shuffle=False, collate_fn=test_collate)
    all_scores, all_tgt = [], []
    with torch.no_grad():
        for batch in te_dl:
            inp = batch["input_ids"].cuda()
            tgt = batch["target"].cuda()
            logits = model(input_ids=inp, attention_mask=(inp != PAD).cuda()).logits[:, -1, :]
            poi_logits = logits[:, poi_tid_arr]  # [B, num_pois]
            if mask_history:
                # SASRec-style protocol on rev.=0 domains: drop already-consumed items
                for bi, hraw in enumerate(batch["hist_raw"]):
                    seen = [x for x in hraw if x < num_pois]
                    if seen:
                        poi_logits[bi, seen] = float("-inf")
            all_scores.append(poi_logits.cpu())
            all_tgt.append(tgt.cpu())
    scores = torch.cat(all_scores, 0)
    targets = torch.cat(all_tgt, 0)
    metrics = rank_metrics(scores, targets, k_list=(1, 5, 10))
    print(f"[{args.city}] LLM4POI-style metrics:", metrics)

    # ---- cold-start subset (identical protocol as honest_eval: train freq <= 5) ----
    from collections import Counter
    freq = Counter(p for seq in train_trajs for p in seq)
    tgts_np = targets.cpu().numpy()
    cold_mask = np.array([freq.get(int(t), 0) <= 5 for t in tgts_np])
    cold = None
    if cold_mask.any():
        cold = {"n": int(cold_mask.sum()),
                "metrics": rank_metrics(scores[cold_mask], targets[cold_mask], k_list=(1, 5, 10))}
        print(f"[{args.city}] cold-start n={cold['n']} metrics:", cold["metrics"])

    # ---- save per-sample scores/targets for later subset analyses ----
    np.savez(os.path.join(HERE, f"llm4poi_{args.city}_scores.npz"),
             scores=scores.float().cpu().numpy(), targets=tgts_np)
    # ---- save LoRA adapter for reuse without retraining ----
    adapter_dir = os.path.join(HERE, f"llm4poi_{args.city}_adapter")
    model.save_pretrained(adapter_dir)
    print("saved adapter", adapter_dir)

    json.dump({"model": "LLM4POI-style (Qwen2.5-3B-Instruct, LoRA)",
               "city": args.city, "k_list": [1, 5, 10],
               "metrics": metrics,
               "cold": cold,
               "n_test": int(len(targets))}, open(args.out, "w"), indent=2)
    print("wrote", args.out)

if __name__ == "__main__":
    main()
