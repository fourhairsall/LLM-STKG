"""只训练 SASRec 并导出【逐样本全候选排名】，供 honest_eval.py 做子集拆分与显著性检验。

动机：用户要求补一个真实 SOTA 序列模型，验证 GRU-STGN 0.0159 是否因「无注意力的 GRU
长序列梯度消失」所致，并提供比 GRU 更强的序列对照——回答「带自注意力的序列模型在
revisit 主导的 Foursquare-NYC 上能到多少、是否仍不敌 History-Freq 平凡基线」。

输出与 baseline_ranks.py 相同的结构（rank_diag.full.SASRec.ranks + metrics），
可直接喂给 honest_eval.py --model "SASRec=文件名.json"。
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_stkg.baselines import SASRec                                # noqa: E402
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
    ap.add_argument("--out", default="sasrec_ranks.json")
    ap.add_argument("--hist_mode", default="user", choices=["trajectory", "user"])
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--maxlen", type=int, default=100,
                    help="SASRec 截断窗口（标准 SASRec 用 50~200）；与协议 seq_len 解耦，"
                         "作为模型超参，不影响训练样本生成协议")
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
    model = SASRec(n_users, num_pois, dim=64, n_layers=2, n_heads=2,
                   maxlen=args.maxlen, dropout=0.1)
    print(f"\n--- SASRec (epochs={args.epochs}, bs={args.bs}) ---", flush=True)
    model.fit(train_samples, epochs=args.epochs, bs=args.bs, device=args.device)

    scores = torch.stack([
        torch.tensor(model.session_predict(h), dtype=torch.float32)
        for _, h, _ in test_samples])
    results = {"SASRec": rank_metrics(scores, tgts)}
    diags = {"SASRec": rank_diag(scores, tgts)}
    print(f"[SASRec] {results['SASRec']}  ({time.time()-t0:.1f}s)", flush=True)

    payload = {"dataset": "Foursquare-NYC (LLM4POI, real)",
               "num_test": len(test_samples), "num_pois": num_pois,
               "epochs": args.epochs, "seed": args.seed,
               "protocol": {"hist_mode": args.hist_mode, "seq_len": args.seq_len,
                            "sasrec_maxlen": args.maxlen,
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
