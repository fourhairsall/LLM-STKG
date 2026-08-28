"""阶段 5 实验 harness：LLM-STKG 与可跑基线在同协议下的正面对比。

用法：
  cd code
  python -m llm_stkg.run_experiment --device cuda --epochs 15 --num_pois 1200 --num_users 3000

数据策略：
  --data_path 指向真实 Foursquare/LLM4POI CSV 时加载真数据；否则生成「schema 完全一致的
  Foursquare-NYC 同构替代数据」用于验证 GPU 全流程（明确标注，非真实 Foursquare）。

输出：
  stage5_results.json  —— 各模型 Recall@5/10、NDCG@5/10、冷启动子集指标、SOTA 文献参考块。
"""
import argparse
import json
import os
import random

from .config import Config
from .train import prepare_splits, train_model, eval_ours_full
from .baselines import build_baselines
from .data.foursquare_loader import load_or_generate


# ---------------- SOTA 文献参考块（明确标注来源与协议） ----------------
# 诚实原则：Ours + 4 基线为本工作「同协议实测」；下列外部 SOTA 为「原论文报告值（其协议）」，
# 精确数字待从对应论文表格誊录，避免编造。GETNext 为广泛引用的锚点值。
SOTA_LITERATURE = {
    "GETNext (KDD 2022)": {
        "dataset": "Foursquare-NYC", "metric": "Recall@10",
        "value": 0.298, "protocol": "原论文报告值（其划分）", "note": "广泛引用锚点；TKY≈0.247",
    },
    "STAN (AAAI 2019)": {
        "dataset": "Foursquare-NYC", "metric": "Recall@10",
        "value": None, "protocol": "原论文报告值（其划分）", "note": "精确值待从原表誊录",
    },
    "CoMaPOI (SIGIR 2025)": {
        "dataset": "Foursquare-NYC/TKY", "metric": "Recall@10",
        "value": None, "protocol": "原论文报告值（其划分）", "note": "LLM 多智能体；相对 GETNext 提升见原文",
    },
    "CaST-POI (2026)": {
        "dataset": "Foursquare-NYC/TKY", "metric": "Recall@10",
        "value": None, "protocol": "原论文报告值（其划分）", "note": "候选条件时空建模；2026 时空 SOTA",
    },
    "RALLM-POI (PRICAI 2025)": {
        "dataset": "Foursquare-NYC/TKY", "metric": "Recall@10",
        "value": None, "protocol": "原论文报告值（其划分）", "note": "检索增强 LLM + 地理重排",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--num_users", type=int, default=3000)
    ap.add_argument("--num_pois", type=int, default=1200)
    ap.add_argument("--max_pois", type=int, default=0)
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="stage5_results.json")
    args = ap.parse_args()

    cfg = Config(
        num_users=args.num_users, num_pois=args.num_pois, seq_len=args.seq_len,
        epochs=args.epochs, device=args.device, seed=args.seed,
        batch_size=256, neg_samples=10, lr=1e-3, num_gnn_layers=2,
        geo_radius_km=1.5, semantic_sim_thr=0.5, covisit_min=3,
    )

    pois, checkins, source = load_or_generate(args.data_path, cfg,
                                              max_pois=args.max_pois or None)
    print(f"[DATA] source={source} | pois={len(pois)} users={len(checkins)}")

    train_samples, val_samples, num_pois = prepare_splits(checkins, cfg.seq_len, cfg.seed)
    num_users = max(u for u, _, _ in train_samples + val_samples) + 1
    print(f"[SPLIT] train={len(train_samples)} val={len(val_samples)} num_pois={num_pois}")

    results = {"source": source, "num_pois": num_pois,
               "num_train": len(train_samples), "num_val": len(val_samples)}

    # ---- Ours: LLM-STKG ----
    print("\n=== 训练 LLM-STKG (ours) ===")
    ours_model, ours_metrics = train_model(cfg, pois, checkins, device=args.device)
    results["LLM-STKG (ours)"] = ours_metrics

    # ---- 基线 ----
    print("\n=== 训练基线 ===")
    baselines = build_baselines(num_users, num_pois, device=args.device)
    for name, model in baselines.items():
        print(f"--- {name} ---")
        model.fit(train_samples, epochs=args.epochs, device=args.device)
        results[name] = model.eval_metrics(val_samples, device=args.device)
        print(f"    {name}: {results[name]}")

    # ---- 冷启动子集（验证用户交互稀疏时仍有效）----
    train_cnt = {}
    for u, _, _ in train_samples:
        train_cnt[u] = train_cnt.get(u, 0) + 1
    cold = [s for s in val_samples if train_cnt.get(s[0], 0) <= 5]
    if cold:
        import torch
        from .evaluate import rank_metrics
        cold_tgt = torch.tensor([t for _, _, t in cold], dtype=torch.long)
        cs = {}
        # ours（复用已训练模型）
        cs["LLM-STKG (ours)"] = eval_ours_full(ours_model, cold, num_pois, cfg, args.device)
        # 基线
        for name, model in baselines.items():
            sc = torch.stack([torch.tensor(model.predict(u, h)) for u, h, _ in cold])
            cs[name] = rank_metrics(sc, cold_tgt, k_list=(5, 10))
        results["cold_start@5interactions"] = {"n": len(cold), **cs}
        print(f"\n[COLD-START] n={len(cold)} subset metrics: {cs}")

    # ---- SOTA 文献块 ----
    results["SOTA_literature"] = SOTA_LITERATURE

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {args.out}")

    # ---- 打印对比表 ----
    print("\n================ 阶段5 对比表 ================")
    print(f"{'模型':<22}{'Recall@5':>10}{'Recall@10':>12}{'NDCG@5':>10}{'NDCG@10':>12}")
    print("-" * 66)
    row = lambda m: (f"{m['Recall@5']:.4f}" if isinstance(m['Recall@5'],(int,float)) else str(m['Recall@5']))
    for k in ["LLM-STKG (ours)", "GRU-STGN", "FPMC", "LightGCN", "BPR-MF"]:
        if k in results:
            m = results[k]
            print(f"{k:<22}{m['Recall@5']:>10.4f}{m['Recall@10']:>12.4f}{m['NDCG@5']:>10.4f}{m['NDCG@10']:>12.4f}")
    print("-" * 66)
    print("外部 SOTA（原论文报告值，协议不同，精确数字待誊录）：")
    for k, v in SOTA_LITERATURE.items():
        val = f"{v['value']:.3f}" if isinstance(v['value'], (int, float)) else "见原论文"
        print(f"  {k:<22}{val:>10}  ({v['dataset']}, {v['metric']})")
    print("=============================================")


if __name__ == "__main__":
    main()
