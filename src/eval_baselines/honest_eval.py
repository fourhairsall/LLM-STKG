"""诚实评测：平凡基线 + 重访/新颖子集拆分 + 配对显著性检验。

为什么需要这个脚本
------------------
1. LLM4POI 的 Foursquare-NYC 划分中 75.7% 的测试目标已在该样本历史里出现，
   "重访"主导了全量 Recall。只报全量指标会把"复读历史"误读成"时空-语义建模能力"。
2. 因此必须同时报：
     (a) History-Freq / History-Recency / Markov-1 三个零学习基线；
     (b) revisit / novel / cold / novel∩cold 四个子集的分别结果；
     (c) ours 与最强平凡基线之间的配对显著性（McNemar + 配对 bootstrap）。
3. 所有指标只依赖"目标的全候选排名"，故可直接从已有 run 的 rank_diag.ranks 复算，
   无需重训——包括把此前被误算为 1/rank 的 NDCG 改回标准 1/log2(rank+1)。

用法
----
  # 单文件（自动读取其中所有模型）
  python honest_eval.py --ours_json _pilot_bs1024_lr4e3_ep30.json

  # 多种子聚合：同一模型给多个 seed 的结果文件，表中报 mean±std
  python honest_eval.py --ours_json baseline_ranks.json \
      --model "LM-STKG+C6=c6_full_s42.json,c6_full_s123.json,c6_full_s777.json" \
      --model "C6 w/o KG channel=c6_nokg_s42.json,c6_nokg_s123.json" \
      --main_model "LM-STKG+C6" \
      --out honest_eval_report.json --md honest_eval_table.md
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_stkg.data.foursquare_loader import load_real_nyc          # noqa: E402
from llm_stkg.evaluate import metrics_from_ranks, target_rank      # noqa: E402
from llm_stkg.trivial_baselines import (                           # noqa: E402
    build_trivial_scores, split_test_subsets)

K_LIST = (1, 5, 10)


# ---------------- 显著性检验 ----------------
def mcnemar_exact(hit_a, hit_b):
    """精确 McNemar（二项检验）：比较两模型在同一批样本上的命中/未命中配对差异。

    hit_a / hit_b : 0-1 数组，同长度、同顺序。
    返回 (b, c, p_two_sided)，b = a对b错 的样本数，c = a错b对 的样本数。
    """
    a = np.asarray(hit_a).astype(bool)
    b_ = np.asarray(hit_b).astype(bool)
    b = int(np.sum(a & ~b_))
    c = int(np.sum(~a & b_))
    n = b + c
    if n == 0:
        return b, c, 1.0
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return b, c, float(min(1.0, 2.0 * tail))


def paired_bootstrap(vals_a, vals_b, n_boot=10000, seed=0):
    """配对 bootstrap：返回 (mean_diff, ci_low, ci_high, p_two_sided)。

    p 值用"差值分布跨过 0 的比例"的双侧版本估计（bootstrap 置换近似）。
    """
    a = np.asarray(vals_a, dtype=float)
    b = np.asarray(vals_b, dtype=float)
    d = a - b
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    obs = float(d.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # 中心化后计算 |boot| >= |obs| 的比例 → 近似双侧 p
    centered = boots - obs
    p = float((np.abs(centered) >= abs(obs)).mean())
    return obs, float(lo), float(hi), p


def per_sample_metric(rank, k, kind):
    r = np.asarray(rank, dtype=float)
    if kind == "recall":
        return (r <= k).astype(float)
    if kind == "ndcg":
        v = np.zeros_like(r)
        m = r <= k
        v[m] = 1.0 / np.log2(r[m] + 1.0)
        return v
    raise ValueError(kind)


def _extract_ranks(path, n_test, prefer="LLM-STKG (ours)"):
    """从一个结果 JSON 里取出目标模型的逐样本排名（找不到就取唯一可用的那个）。"""
    if not os.path.exists(path):
        print(f"[warn] 文件不存在，跳过: {path}")
        return None
    d = json.load(open(path, encoding="utf-8"))
    full = d.get("rank_diag", {}).get("full", {})
    cand = full.get(prefer, {}).get("ranks")
    if not cand:
        avail = [m for m, v in full.items() if v.get("ranks")]
        if len(avail) != 1:
            print(f"[warn] {os.path.basename(path)} 中未找到 '{prefer}'，可用={avail}，跳过")
            return None
        cand = full[avail[0]]["ranks"]
    if len(cand) != n_test:
        print(f"[warn] {os.path.basename(path)} ranks 长度 {len(cand)} != {n_test}，跳过")
        return None
    return np.asarray(cand, dtype=float)


def agg(vals):
    """多种子聚合 → (mean, std)。单种子时 std=0。"""
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def fmt(mean, std, prec=4):
    return f"{mean:.{prec}f}" if std == 0 else f"{mean:.{prec}f}±{std:.{prec}f}"


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default=None)
    ap.add_argument("--ours_json", nargs="+", required=True,
                    help="包含 rank_diag.full.<model>.ranks 的结果 JSON（可多个，按模型名合并）")
    ap.add_argument("--model", action="append", default=[],
                    help="显式命名的模型（可重复）：'显示名=f1.json,f2.json,...'，"
                         "多个文件视为同一模型的不同随机种子，表中报 mean±std")
    ap.add_argument("--main_model", default=None,
                    help="做配对显著性检验的主模型显示名；缺省=自动找 ours/LM-STKG")
    ap.add_argument("--sig_against", default=None,
                    help="逗号分隔的对照模型名；缺省=三个平凡基线 + LightGCN + BPR-MF（若存在）")
    ap.add_argument("--cold_thr", type=int, default=5)
    ap.add_argument("--out", default="honest_eval_report.json")
    ap.add_argument("--md", default="honest_eval_table.md")
    args = ap.parse_args()

    pois, checkins, test_samples, num_pois, stats, _ = load_real_nyc(
        args.processed_dir, cold_poi_ratio=0.0)
    print(f"[data] num_pois={num_pois} n_test={len(test_samples)} "
          f"n_train_traj={len(checkins)}")

    tgts = torch.tensor([int(t) for _, _, t in test_samples], dtype=torch.long)
    n_test = len(test_samples)

    # ranks[显示名] = [seed0_ranks, seed1_ranks, ...]（单种子时列表长度 1）
    ranks = {}

    # ---- 1. 平凡基线的逐样本排名（零参数，无种子）----
    mats = build_trivial_scores(test_samples, num_pois, checkins)
    for name, S in mats.items():
        r = target_rank(S, tgts).numpy()
        ranks[name] = [r]
        print(f"[trivial] {name}: R@10={float((r<=10).mean()):.4f} "
              f"median_rank={float(np.median(r)):.1f}")

    # ---- 2. 从已有 run 读取学习型模型的排名（自动模式：文件里有几个模型收几个）----
    for jf in args.ours_json:
        if not os.path.exists(jf):
            print(f"[warn] 结果文件不存在，跳过: {jf}")
            continue
        d = json.load(open(jf, encoding="utf-8"))
        full = d.get("rank_diag", {}).get("full", {})
        for mname, diag in full.items():
            rr = diag.get("ranks")
            if not rr:
                continue
            if len(rr) != n_test:
                print(f"[warn] {jf}:{mname} ranks 长度 {len(rr)} != "
                      f"n_test {n_test}，跳过（协议不一致）")
                continue
            key = mname if mname not in ranks else f"{mname} [{os.path.basename(jf)}]"
            ranks[key] = [np.asarray(rr, dtype=float)]
            print(f"[loaded] {key} <- {os.path.basename(jf)}")

    # ---- 2b. 显式命名的多种子模型（覆盖同名自动条目）----
    for spec in args.model:
        if "=" not in spec:
            print(f"[warn] --model 格式应为 '名字=f1.json,f2.json'，忽略: {spec}")
            continue
        disp, files = spec.split("=", 1)
        disp = disp.strip()
        seeds = [r for r in (_extract_ranks(f.strip(), n_test)
                             for f in files.split(",") if f.strip()) if r is not None]
        if not seeds:
            print(f"[warn] 模型 '{disp}' 无可用种子文件，跳过")
            continue
        ranks[disp] = seeds
        r10 = [float((s <= 10).mean()) for s in seeds]
        print(f"[model] {disp}: {len(seeds)} seed(s), R@10={fmt(*agg(r10))}")

    if not ranks:
        raise SystemExit("没有任何可用排名，退出。")

    # ---- 3. 子集切分 ----
    subsets = split_test_subsets(test_samples, checkins, cold_thr=args.cold_thr)
    hist_len = [len(h) for _, h, _ in test_samples]
    print("\n[subset sizes] " + ", ".join(f"{k}={len(v)}" for k, v in subsets.items()))

    # ---- 4. 逐子集指标（多种子 → 逐种子算指标再聚合 mean±std）----
    table = {}
    for sname, idx in subsets.items():
        if not idx:
            continue
        ii = np.asarray(idx)
        table[sname] = {"n": len(idx)}
        for mname, seeds in ranks.items():
            per_seed = [metrics_from_ranks(torch.tensor(r[ii], dtype=torch.float32),
                                           k_list=K_LIST) for r in seeds]
            entry = {"n_seeds": len(seeds)}
            for k in per_seed[0]:
                mu, sd = agg([p[k] for p in per_seed])
                entry[k] = round(mu, 4)
                if len(seeds) > 1:
                    entry[k + "_std"] = round(sd, 4)
            table[sname][mname] = entry

    # ---- 5. 显著性：主模型（seed 0）vs 平凡基线与神经基线 ----
    ours_key = args.main_model or next(
        (k for k in ranks if "ours" in k.lower() or k.lower().startswith("lm-stkg")
         or k.lower().startswith("llm-stkg")), None)
    sig = {}
    if ours_key and ours_key in ranks:
        if args.sig_against:
            trivial_keys = [k.strip() for k in args.sig_against.split(",")
                            if k.strip() in ranks]
        else:
            trivial_keys = [k for k in ("History-Freq (HF)", "History-Recency (HR)",
                                        "Markov-1 (MC1)", "LightGCN", "BPR-MF")
                            if k in ranks]
        print(f"[sig] 主模型={ours_key}（用 seed 0 的逐样本排名）"
              f" vs {trivial_keys}")
        for sname, idx in subsets.items():
            if not idx:
                continue
            ii = np.asarray(idx)
            ro = ranks[ours_key][0][ii]
            sig[sname] = {"n": len(idx)}
            for tk in trivial_keys:
                rt = ranks[tk][0][ii]
                ha = per_sample_metric(ro, 10, "recall")
                hb = per_sample_metric(rt, 10, "recall")
                b, c, p_mc = mcnemar_exact(ha, hb)
                na = per_sample_metric(ro, 10, "ndcg")
                nb = per_sample_metric(rt, 10, "ndcg")
                dmean, lo, hi, p_bs = paired_bootstrap(na, nb)
                sig[sname][tk] = {
                    "R@10_ours": round(float(ha.mean()), 4),
                    "R@10_baseline": round(float(hb.mean()), 4),
                    "mcnemar_b_ours_only": b,
                    "mcnemar_c_baseline_only": c,
                    "mcnemar_p": round(p_mc, 6),
                    "ndcg10_diff_mean": round(dmean, 4),
                    "ndcg10_diff_ci95": [round(lo, 4), round(hi, 4)],
                    "bootstrap_p": round(p_bs, 4),
                }

    # ---- 6. 协议画像（写进论文 §5.1，让读者自行判断评测难度）----
    profile = {
        "n_test": len(test_samples),
        "revisit_ratio": round(len(subsets["revisit"]) / len(test_samples), 4),
        "novel_ratio": round(len(subsets["novel"]) / len(test_samples), 4),
        "history_len_mean": round(float(np.mean(hist_len)), 2),
        "history_len_median": float(np.median(hist_len)),
        "history_len_p90": float(np.percentile(hist_len, 90)),
        "distinct_poi_in_history_mean": round(float(np.mean(
            [len(set(h)) for _, h, _ in test_samples])), 2),
        "cold_thr": args.cold_thr,
        "subset_sizes": {k: len(v) for k, v in subsets.items()},
    }

    payload = {"protocol_profile": profile, "metrics_by_subset": table,
               "significance_vs_trivial": sig,
               "note": ("NDCG 已改为标准定义 1/log2(rank+1)；此前代码中的 1/rank "
                        "实为截断 MRR，两者均在此报告中给出（MRR@k）。")}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {args.out}")

    # ---- 7. Markdown 表 ----
    lines = ["# Honest evaluation: trivial baselines & revisit/novel split", "",
             f"- n_test = {profile['n_test']}, revisit = {profile['revisit_ratio']:.1%}, "
             f"novel = {profile['novel_ratio']:.1%}",
             f"- mean history length = {profile['history_len_mean']} "
             f"(median {profile['history_len_median']}, p90 {profile['history_len_p90']})",
             f"- mean distinct POIs in history = {profile['distinct_poi_in_history_mean']}",
             ""]
    order = [k for k in ("all", "revisit", "novel", "cold", "revisit_cold", "novel_cold")
             if k in table]
    mnames = list(ranks.keys())
    for sname in order:
        lines += [f"## subset = {sname} (n = {table[sname]['n']})", "",
                  "| Method | R@1 | R@5 | R@10 | NDCG@5 | NDCG@10 | MRR |",
                  "|---|---|---|---|---|---|---|"]
        for m in mnames:
            v = table[sname].get(m)
            if not v:
                continue
            def cell(key):
                return fmt(v[key], v.get(key + "_std", 0.0))
            lines.append(f"| {m} | {cell('Recall@1')} | {cell('Recall@5')} | "
                         f"{cell('Recall@10')} | {cell('NDCG@5')} | "
                         f"{cell('NDCG@10')} | {cell('MRR')} |")
        lines.append("")
    if sig:
        lines += ["## Paired significance: ours vs trivial baselines", "",
                  "| Subset | Baseline | ours R@10 | base R@10 | McNemar b/c | p | "
                  "ΔNDCG@10 [95% CI] | boot p |", "|---|---|---|---|---|---|---|---|"]
        for sname in order:
            for tk, s in sig.get(sname, {}).items():
                if not isinstance(s, dict):
                    continue
                lines.append(
                    f"| {sname} | {tk} | {s['R@10_ours']:.4f} | {s['R@10_baseline']:.4f} | "
                    f"{s['mcnemar_b_ours_only']}/{s['mcnemar_c_baseline_only']} | "
                    f"{s['mcnemar_p']:.4g} | {s['ndcg10_diff_mean']:+.4f} "
                    f"[{s['ndcg10_diff_ci95'][0]:+.4f}, {s['ndcg10_diff_ci95'][1]:+.4f}] | "
                    f"{s['bootstrap_p']:.4g} |")
        lines.append("")
    with open(args.md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {args.md}")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
