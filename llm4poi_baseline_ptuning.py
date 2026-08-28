"""
LLM4POI-style generative baseline -- P-TUNING v2 (PrefixTuning) variant.

Faithful P-tuning-v2 reproduction (replaces the LoRA used in
llm4poi_baseline.py, whose own docstring admitted "not a faithful
P-tuning-v2 reproduction"). Each POI is a special token <POI_i>; a
trajectory is a sequence of POI tokens and the model is trained with
next-POI-token prediction. At evaluation we take the logits over ALL POI
tokens at the last position and rank the true target among all POIs --
exactly the full-candidate protocol used by LLM4POI and identical to the
protocol of our LLM-STKG (we reuse llm_stkg.evaluate.rank_metrics).

Key design points (kept identical to llm4poi_baseline.py for a fair,
controlled comparison):
  * POI token embeddings are INITIALISED from BGE semantic vectors
    (hidden = [bge | 0]); this injects the strongest semantic prior
    available, mirroring LLM4POI's geospatial-embedding initialisation,
    so any under-performance is not due to a weak init strawman.
  * The base LM is FROZEN; only the P-tuning prefix (virtual tokens +
    prompt-encoder MLP) and the POI input embeddings are trained.

Hardware adaptation for 7B on a 12 GB GPU:
  * --load_in_8bit quantises the base LM to int8, but keeps
    embed_tokens and lm_head in full precision (llm_int8_skip_modules)
    so the BGE semantic seeding above still works and the output
    projection stays exact.
  * --model_dir lets you point at Llama-2-7B (or any causal LM). On the
    local Qwen2.5-3B-Instruct you can run WITHOUT --load_in_8bit.

Usage examples:
  # smoke / fair 3B variant (P-tuning, no quant needed)
  python llm4poi_baseline_ptuning.py --city nyc
  # 7B on 12 GB (P-tuning + 8-bit)
  python llm4poi_baseline_ptuning.py --city nyc \
      --model_dir /path/to/Llama-2-7b-hf --load_in_8bit \
      --batch 1 --grad_accum 16 --epochs 3
"""
import os, sys, json, argparse, math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          get_linear_schedule_with_warmup, BitsAndBytesConfig)
from torch.optim import AdamW
from peft import LoraConfig, PrefixTuningConfig, get_peft_model
from peft import prepare_model_for_kbit_training

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(HERE, "..", ".."))  # .../专利写作/2026年7月
sys.path.insert(0, HERE)
from llm_stkg.evaluate import rank_metrics

MAX_LEN = 64
SEED = 42

def _model_tag(model_dir):
    return os.path.basename(os.path.normpath(model_dir))

class TrajDataset(Dataset):
    def __init__(self, trajs, poi_tid, max_len=MAX_LEN, aug_max=0):
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
    ap.add_argument("--peft", choices=["ptuning", "lora"], default="ptuning",
                    help="ptuning = P-tuning v2 (PrefixTuning); lora = legacy LoRA")
    ap.add_argument("--num_virtual_tokens", type=int, default=20,
                    help="P-tuning v2 prefix length")
    ap.add_argument("--encoder_hidden_size", type=int, default=512,
                    help="P-tuning v2 prompt-encoder MLP hidden size")
    ap.add_argument("--load_in_8bit", action="store_true",
                    help="8-bit quant base LM (required for 7B on a 12 GB GPU); "
                         "embed_tokens/lm_head kept in fp16 so BGE seeding survives")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="4-bit quant base LM (more headroom to unfreeze the whole "
                         "lm_head so POI logits stop collapsing to a constant)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max_gsteps", type=int, default=0,
                    help="If >0, hard cap on optimizer steps (alternative to --max_steps "
                         "which counts micro-batches). Used for segmented training.")
    ap.add_argument("--resume_adapter", type=str, default=None,
                    help="Path to a previously saved PEFT adapter dir; if set, the model "
                         "is loaded from it (continuing training) instead of creating a fresh "
                         "PEFT model. Used to chain short training segments around the ~5-min "
                         "background-process lifetime limit.")
    ap.add_argument("--save_every", type=int, default=0,
                    help="If >0, save the adapter + a partial JSON every N optimizer steps "
                         "(checkpoint for resuming short segments).")
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="P-tuning trains few params; a higher LR than LoRA is typical")
    ap.add_argument("--max_steps", type=int, default=100000)
    ap.add_argument("--aug_max", type=int, default=0,
                    help="sliding-window augmentation cap per trajectory (0=off)")
    ap.add_argument("--no_bge", action="store_true",
                    help="disable BGE semantic seeding (causal ablation)")
    ap.add_argument("--no_grad_ckpt", action="store_true",
                    help="disable gradient checkpointing")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(HERE, f"llm4poi_{args.peft}_{_model_tag(args.model_dir)}_{args.city}.json")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    global PAD

    city = args.city
    if args.data_root:
        processed_dir = args.data_root
    else:
        _cross_map = {"steam": "steam", "gowalla": "gowalla", "ml1m": "ml-1m",
                      "steam200k": "steam200k", "amazon_beauty": "amazon_beauty"}
        _sub = _cross_map.get(city, "real_foursquare_%s" % city)
        processed_dir = os.path.join(WORKSPACE, "data", _sub, "processed")
    # ---- data loading (per-domain formats) ----
    bge_cache = None
    pois = None
    if city in ("nyc", "tky"):
        raw_train = json.load(open(os.path.join(processed_dir, "train_trajs.json"), encoding="utf-8"))
        train_trajs = [t["pois"] for t in raw_train]
        test_pairs = json.load(open(os.path.join(processed_dir, "test_pairs.json"), encoding="utf-8"))
        bge_cache = os.path.join(HERE, "poi_bge_emb%s.npy" % ("" if city == "nyc" else "_tky"))
        num_pois = int(np.load(bge_cache).shape[0])
    else:
        raw_train = json.load(open(os.path.join(processed_dir, "train_checkins.json"), encoding="utf-8"))
        train_trajs = [t["pois"] for t in raw_train]
        test_pairs = json.load(open(os.path.join(processed_dir, "test_samples.json"), encoding="utf-8"))
        pois = json.load(open(os.path.join(processed_dir, "pois.json"), encoding="utf-8"))
        num_pois = int(len(pois))
        _bge_map = {"ml1m": "poi_bge_emb_ml1m.npy",
                    "steam200k": "poi_bge_emb_steam200k.npy",
                    "amazon_beauty": "poi_bge_emb_amazonbeauty.npy"}
        bge_cache = os.path.join(HERE, _bge_map[city]) if city in _bge_map else None
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
          f"mask_history={'ON' if mask_history else 'OFF'} model={_model_tag(args.model_dir)} peft={args.peft}")
    if pois is not None and len(pois) > 0:
        print(f"[{city}] poi text sample: {str(pois[0].get('text', ''))[:70]!r}")

    # OpenLLaMA ships a SentencePiece tokenizer.model; transformers>=5.x may
    # mis-detect it as a tiktoken file and crash on the fast path. Fall back to
    # the slow SentencePiece tokenizer (needs sentencepiece, already present).
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    PAD = tokenizer.pad_token_id

    # add POI special tokens
    poi_tokens = [f"<POI_{i}>" for i in range(num_pois)]
    tokenizer.add_tokens(poi_tokens)
    base = tokenizer.convert_tokens_to_ids(poi_tokens[0])

    # ---- base LM load (optional 8-bit) ----
    # P-tuning v2 (PEFT prefix tuning) is incompatible with the default SDPA
    # attention backend in transformers >= 5.x: SDPA receives the un-prefixed
    # attention_mask while the hidden states are already prefix-extended, causing
    # a query/key length mismatch ([4,1,14,34] vs [4,16,14,14]). Force the eager
    # backend for the P-tuning path. LoRA is unaffected and keeps SDPA.
    quant_cfg = None
    attn_impl = "eager" if args.peft == "ptuning" else "sdpa"
    if args.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["embed_tokens", "lm_head"])
    elif args.load_in_8bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["embed_tokens", "lm_head"])
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        quantization_config=quant_cfg, attn_implementation=attn_impl,
        device_map="auto" if (args.load_in_8bit or args.load_in_4bit) else None)
    if not args.load_in_8bit:
        model = model.cuda()
    # For k-bit (8/4) base models, prepare_model_for_kbit_training correctly casts the
    # lm_head and input/output embeddings to trainable fp16/bf16 (the standard PEFT
    # recipe). Without this, manually setting requires_grad on the quantized lm_head is
    # a no-op and ALL POI logits collapse to a constant (per-row std ~1e-4 -> R@10=0.0),
    # even with LoRA -- exactly the failure seen vs the unquantized 3B LoRA baseline
    # (R@10=0.2274) which has a naturally trainable lm_head.
    if args.load_in_8bit or args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
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
            w[base:base + num_pois] = w[base:base + num_pois].to(w.dtype) * 0.1
    model.get_input_embeddings().weight.requires_grad_(True)  # learn POI reps

    # ---- PEFT: P-tuning v2 (default) or legacy LoRA ----
    if args.peft == "ptuning":
        peft_cfg = PrefixTuningConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=args.num_virtual_tokens,
            prefix_projection=True,
            encoder_hidden_size=args.encoder_hidden_size)
    else:
        peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                              "gate_proj", "up_proj", "down_proj", "lm_head"],
                              task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, peft_cfg)
    if args.resume_adapter:
        # Continue from a saved adapter (short-segment chaining).
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume_adapter)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # ---- train ----
    tr_ds = TrajDataset(train_trajs, poi_tid, aug_max=args.aug_max)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    print(f"[{city}] train samples after aug={len(tr_ds)}")
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    total_opt = args.epochs * (len(tr_dl) // args.grad_accum + 1)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=max(1, total_opt // 20), num_training_steps=total_opt)
    model.train()
    mb = 0; gstep = 0
    for ep in range(args.epochs):
        for batch in tr_dl:
            batch = {k: v.cuda() for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            if not torch.isfinite(loss):
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
                if args.save_every and gstep % args.save_every == 0:
                    ckpt = os.path.join(HERE, f"llm4poi_{_model_tag(args.model_dir)}_{args.city}_{args.peft}_ckpt{gstep}")
                    model.save_pretrained(ckpt)
                    print(f"[{args.city}] checkpoint saved -> {ckpt} (gstep={gstep})", flush=True)
            if mb >= args.max_steps:
                break
            if args.max_gsteps and gstep >= args.max_gsteps:
                break
        if mb % args.grad_accum != 0:
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
            poi_logits = logits[:, poi_tid_arr]
            if mask_history:
                for bi, hraw in enumerate(batch["hist_raw"]):
                    seen = [x for x in hraw if x < num_pois]
                    if seen:
                        poi_logits[bi, seen] = float("-inf")
            all_scores.append(poi_logits.cpu())
            all_tgt.append(tgt.cpu())
    scores = torch.cat(all_scores, 0)
    targets = torch.cat(all_tgt, 0)
    metrics = rank_metrics(scores, targets, k_list=(1, 5, 10))
    print(f"[{args.city}] LLM4POI-style ({_model_tag(args.model_dir)} + {args.peft}) metrics:", metrics)

    from collections import Counter
    freq = Counter(p for seq in train_trajs for p in seq)
    tgts_np = targets.cpu().numpy()
    cold_mask = np.array([freq.get(int(t), 0) <= 5 for t in tgts_np])
    cold = None
    if cold_mask.any():
        cold = {"n": int(cold_mask.sum()),
                "metrics": rank_metrics(scores[cold_mask], targets[cold_mask], k_list=(1, 5, 10))}
        print(f"[{args.city}] cold-start n={cold['n']} metrics:", cold["metrics"])

    np.savez(os.path.join(HERE, f"llm4poi_{args.peft}_{_model_tag(args.model_dir)}_{args.city}_scores.npz"),
             scores=scores.float().cpu().numpy(), targets=tgts_np)
    adapter_dir = os.path.join(HERE, f"llm4poi_{_model_tag(args.model_dir)}_{args.city}_{args.peft}_adapter")
    model.save_pretrained(adapter_dir)
    print("saved adapter", adapter_dir)

    json.dump({"model": f"LLM4POI-style ({_model_tag(args.model_dir)} + {args.peft})",
               "base_model": _model_tag(args.model_dir),
               "peft": args.peft,
               "load_in_8bit": bool(args.load_in_8bit),
               "load_in_4bit": bool(args.load_in_4bit),
               "city": args.city, "k_list": [1, 5, 10],
               "metrics": metrics,
               "cold": cold,
               "n_test": int(len(targets))}, open(args.out, "w"), indent=2)
    print("wrote", args.out)

if __name__ == "__main__":
    main()
