"""只训练 eSASRec (Tikhonovich et al., RecSys'25) 并导出【逐样本全候选排名】。

eSASRec = SASRec 训练目标 + LiGR Transformer 层(RMSNorm+SwiGLU) + Sampled Softmax Loss。
本脚本与 sasrec_ranks.py 完全对齐协议（hist_mode=user / seq_len=200 / maxlen 解耦为模型超参），
仅把模型换成 eSASRec，从而公平回答「2025 SOTA 序列增强在 replay 主导数据上能否超越 SASRec、

能否逼近 ours(replay + 行为先验 C6)」。

输出结构与 sasrec_ranks.py 一致（rank_diag.full.eSASRec.ranks + metrics），
可直接喂给 honest_eval.py --ours_json ... esasrec_ranks_s42.json
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_stkg.baselines import eSASRec                              # noqa: E402
from llm_stkg.config import Config                                  # noqa: E402
from llm_stkg.data.foursquare_loader import load_real_nyc           # noqa: E402
from llm_stkg.evaluate import rank_metrics, rank_diag               # noqa: E402
from llm_stkg.train import _build_samples                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="esasrec_ranks.json")
    ap.add_argument("--hist_mode", default="user", choices=["trajectory", "user"])
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--maxlen", type=int, default=100,
                    help="eSASRec 截断窗口（与 SASRec 对齐）")
    ap.add_argument("--num_neg", type=int, default=100,
                    help="Sampled Softmax Loss 的均匀随机负样本数")
    ap.add_argument("--loss", default="sampled_softmax",
                    choices=["sampled_softmax", "ce"],
                    help="sampled_softmax=eSASRec 原版损失；ce=仅换 LiGR 架构、"
                         "保留与 SASRec 相同的 plain CE（隔离架构 vs 损失的贡献）")
    ap.add_argument("--max_train", type=int, default=None,
                    help="仅用前 N 个训练样本（pilot 用）；缺省=全量")
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    pois, checkins, test_samples, num_pois, stats, _ = load_real_nyc(
        args.processed_dir, cold_poi_ratio=0.0)
    users = list({u for u, _ in checkins})
    n_users = max(users) + 1
    train_samples = _build_samples(checkins, args.seq_len, set(users),
                                   hist_mode=args.hist_mode)
    if args.max_train:
        train_samples = train_samples[:args.max_train]
    hl = sum(len(h) for _, h, _ in train_samples) / max(1, len(train_samples))
    rv = sum(1 for _, h, t in train_samples if t in set(h)) / max(1, len(train_samples))
    print(f"[data] num_pois={num_pois} n_test={len(test_samples)} "
          f"n_train_samples={len(train_samples)} n_users={n_users}")
    print(f"[samples] hist_mode={args.hist_mode} seq_len={args.seq_len} "
          f"hist_len_mean={hl:.1f} revisit_ratio={rv:.4f}")

    tgts = torch.tensor([int(t) for _, _, t in test_samples], dtype=torch.long)
    model = eSASRec(n_users, num_pois, dim=64, n_layers=2, n_heads=2,
                    maxlen=args.maxlen, dropout=0.1, num_neg=args.num_neg,
                    loss_mode=args.loss)
    key = "eSASRec" if args.loss == "sampled_softmax" else "eSASRec-CE"
    print(f"\n--- eSASRec (loss={args.loss}, epochs={args.epochs}, bs={args.bs}, "
          f"num_neg={args.num_neg}) ---", flush=True)
    model.fit(train_samples, epochs=args.epochs, bs=args.bs, device=args.device)

    scores = torch.stack([
        torch.tensor(model.session_predict(h), dtype=torch.float32)
        for _, h, _ in test_samples])
    results = {key: rank_metrics(scores, tgts)}
    diags = {key: rank_diag(scores, tgts)}
    print(f"[{key}] {results[key]}  ({time.time()-t0:.1f}s)", flush=True)

    payload = {"dataset": "Foursquare-NYC (LLM4POI, real)",
               "model": f"eSASRec (Tikhonovich et al., RecSys'25, loss={args.loss})",
               "num_test": len(test_samples), "num_pois": num_pois,
               "epochs": args.epochs, "seed": args.seed,
               "protocol": {"hist_mode": args.hist_mode, "seq_len": args.seq_len,
                            "sasrec_maxlen": args.maxlen,
                            "num_neg": args.num_neg,
                            "n_train_samples": len(train_samples),
                            "hist_len_mean": round(hl, 1),
                            "revisit_ratio": round(rv, 4),
                            "bs": args.bs, "max_train": args.max_train},
               "results": results, "rank_diag": {"full": diags}}
    json.dump(payload, open(args.out, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\n[saved] {args.out}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
