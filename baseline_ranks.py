"""只训练 4 个学习型基线并导出【逐样本全候选排名】，供 honest_eval.py 做子集拆分。

动机：此前所有 head_to_head run 都只保存了基线的聚合指标，没保存 rank_diag.ranks，
因此无法回答本工作最关键的问题——「在 novel（target ∉ history）子集上，
LightGCN / BPR-MF / FPMC / GRU 各自能做到多少？」。没有这个对照，
就无法判断 KG+语义带来的 novel 子集提升是否真实存在。

输出与 head_to_head 相同的结构（rank_diag.full.<model>.ranks），可直接喂给 honest_eval.py。
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_stkg.baselines import build_baselines                      # noqa: E402
from llm_stkg.config import Config                                  # noqa: E402
from llm_stkg.data.foursquare_loader import load_real_nyc           # noqa: E402
from llm_stkg.evaluate import rank_metrics, rank_diag               # noqa: E402
from llm_stkg.train import _build_samples                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="baseline_ranks.json")
    # 训练样本协议必须与 ours 完全一致，否则对照不公平。
    # trajectory（旧默认）只在单条会话内取前缀，每条轨迹的首个签到永远不是训练目标，
    # 全库因此少 9975 个正样本对（72206 vs 82181），对 FPMC 这类只看 last item 的
    # 序列基线尤其不利——跨会话转移在旧协议下根本不可见。
    ap.add_argument("--hist_mode", default="user", choices=["trajectory", "user"])
    ap.add_argument("--seq_len", type=int, default=200)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    pois, checkins, test_samples, num_pois, stats, _ = load_real_nyc(
        args.processed_dir, cold_poi_ratio=0.0)
    users = list({u for u, _ in checkins})
    n_users = max(users) + 1
    # 与 head_to_head 完全一致的样本构造（同 hist_mode、同 seq_len、同全量训练用户）
    train_samples = _build_samples(checkins, args.seq_len, set(users),
                                   hist_mode=args.hist_mode)
    hl = sum(len(h) for _, h, _ in train_samples) / max(1, len(train_samples))
    rv = sum(1 for _, h, t in train_samples if t in set(h)) / max(1, len(train_samples))
    print(f"[data] num_pois={num_pois} n_test={len(test_samples)} "
          f"n_train_samples={len(train_samples)} n_users={n_users}")
    print(f"[samples] hist_mode={args.hist_mode} seq_len={args.seq_len} "
          f"hist_len_mean={hl:.1f} revisit_ratio={rv:.4f} (测试端为 143.2 / 0.7574)")

    tgts = torch.tensor([int(t) for _, _, t in test_samples], dtype=torch.long)
    results, diags = {}, {}

    # 热度基线（无需训练）
    from collections import Counter
    freq = Counter(p for _, seq in checkins for p in seq)
    pop = torch.zeros(num_pois, dtype=torch.float32)
    for p, c in freq.items():
        if 0 <= p < num_pois:
            pop[p] = float(c)
    S_pop = pop.unsqueeze(0).repeat(len(test_samples), 1)
    results["Popularity (Pop)"] = rank_metrics(S_pop, tgts)
    diags["Popularity (Pop)"] = rank_diag(S_pop, tgts)
    print(f"[Pop] {results['Popularity (Pop)']}")

    baselines = build_baselines(n_users, num_pois, device=args.device)
    for name, m in baselines.items():
        print(f"\n--- {name} (epochs={args.epochs}) ---", flush=True)
        m.fit(train_samples, epochs=args.epochs, device=args.device)
        scores = torch.stack([
            torch.tensor(m.session_predict(h), dtype=torch.float32)
            for _, h, _ in test_samples])
        results[name] = rank_metrics(scores, tgts)
        diags[name] = rank_diag(scores, tgts)
        print(f"[{name}] {results[name]}", flush=True)
        # 增量落盘，避免长跑中断全丢
        json.dump({"dataset": "Foursquare-NYC (LLM4POI, real)",
                   "num_test": len(test_samples), "num_pois": num_pois,
                   "epochs": args.epochs, "seed": args.seed,
                   "protocol": {"hist_mode": args.hist_mode, "seq_len": args.seq_len,
                                "n_train_samples": len(train_samples),
                                "hist_len_mean": round(hl, 1),
                                "revisit_ratio": round(rv, 4)},
                   "results": results, "rank_diag": {"full": diags}},
                  open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
