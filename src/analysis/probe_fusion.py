"""探针：LM-STKG 在「历史频次」这一平凡基线之上，是否还携带增量信息？

方法：倒数排名融合（Reciprocal Rank Fusion, RRF）
    s_fuse(p) = w * 1/(k + rank_ours(p)) + (1-w) * 1/(k + rank_HF(p))
RRF 只需要排名、不需要原始分数，因此可以直接复用已保存的 rank_diag.ranks，
无需重训任何模型即可判断两路信号是否互补。

判据：
  - 若 fusion 在全量 R@10 上显著高于 HF 单独 → ours 携带 HF 没有的信息，
    "重访先验 + KG 语义"的混合模型路线成立；
  - 若 fusion ≈ HF → ours 的信号被 HF 完全覆盖，模型无独立价值，必须改问题定义。

⚠️ 限制：这里只能对【目标 POI 自身】的排名做融合验证，无法重排整个候选集
（缺少非目标 POI 的排名）。因此本探针给出的是**乐观上界的粗估**，
真正结论仍需在完整分数矩阵上做融合。见文末打印的说明。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_stkg.data.foursquare_loader import load_real_nyc          # noqa: E402
from llm_stkg.evaluate import target_rank                          # noqa: E402
from llm_stkg.trivial_baselines import (                           # noqa: E402
    build_trivial_scores, split_test_subsets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_json", default="_pilot_bs1024_lr4e3_ep30.json")
    ap.add_argument("--processed_dir", default=None)
    ap.add_argument("--rrf_k", type=int, default=60)
    ap.add_argument("--out", default="probe_fusion.json")
    args = ap.parse_args()

    pois, checkins, test_samples, num_pois, stats, _ = load_real_nyc(
        args.processed_dir, cold_poi_ratio=0.0)
    tgts = torch.tensor([int(t) for _, _, t in test_samples], dtype=torch.long)
    subsets = split_test_subsets(test_samples, checkins)

    # HF 完整分数矩阵（可重排全候选）
    S_hf = build_trivial_scores(test_samples, num_pois, checkins)["History-Freq (HF)"]

    d = json.load(open(args.ours_json, encoding="utf-8"))
    r_ours = np.asarray(
        d["rank_diag"]["full"]["LLM-STKG (ours)"]["ranks"], dtype=float)
    r_hf = target_rank(S_hf, tgts).numpy().astype(float)

    # --- 1. 互补性诊断：两模型各自"独占命中"的样本数 ---
    rep = {"rrf_k": args.rrf_k, "subsets": {}}
    for sname, idx in subsets.items():
        if not idx:
            continue
        ii = np.asarray(idx)
        ho = r_ours[ii] <= 10
        hh = r_hf[ii] <= 10
        both = int(np.sum(ho & hh))
        only_o = int(np.sum(ho & ~hh))
        only_h = int(np.sum(~ho & hh))
        neither = int(np.sum(~ho & ~hh))
        # 若 only_o 显著 > 0，说明 ours 覆盖了 HF 覆盖不到的样本 → 融合有上界收益
        union = (both + only_o + only_h) / len(ii)
        rep["subsets"][sname] = {
            "n": len(ii),
            "R@10_ours": round(float(ho.mean()), 4),
            "R@10_HF": round(float(hh.mean()), 4),
            "both_hit": both, "ours_only": only_o, "hf_only": only_h,
            "neither": neither,
            "oracle_union_R@10": round(float(union), 4),
            "complementarity": round(only_o / max(1, both + only_o + only_h), 4),
        }

    # --- 2. 真实可实现的融合：对 HF 分数矩阵做 RRF 重排，需要 ours 对全候选的排名 ---
    #     我们只有目标的排名，故这里给出「目标侧 RRF 分数」与「HF 侧同分位竞争者」的
    #     保守近似：把 ours 的排名视为该目标在 ours 下的位次，用 RRF 公式合成后，
    #     与 HF 单独排序做 Kendall 方向一致性对比。仅作趋势判断，不作为论文数字。
    rrf_tgt = 1.0 / (args.rrf_k + r_ours) + 1.0 / (args.rrf_k + r_hf)
    hf_tgt = 1.0 / (args.rrf_k + r_hf)
    rep["rrf_target_score_gain_mean"] = round(
        float(np.mean(rrf_tgt - hf_tgt) / (np.mean(hf_tgt) + 1e-12)), 4)

    json.dump(rep, open(args.out, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print(f"{'subset':<14}{'n':>6}{'ours':>9}{'HF':>9}{'both':>7}"
          f"{'oursOnly':>10}{'hfOnly':>8}{'union':>9}{'compl.':>9}")
    for s, v in rep["subsets"].items():
        print(f"{s:<14}{v['n']:>6}{v['R@10_ours']:>9.4f}{v['R@10_HF']:>9.4f}"
              f"{v['both_hit']:>7}{v['ours_only']:>10}{v['hf_only']:>8}"
              f"{v['oracle_union_R@10']:>9.4f}{v['complementarity']:>9.4f}")
    print(f"\n[saved] {args.out}")
    print("\n解读：oracle_union 是「二选一完美路由」的 R@10 上界；"
          "ours_only 是 ours 独占命中数。\nours_only 越大 → 混合模型越有空间；"
          "若 ours_only≈0 则 ours 的信号已被 HF 完全包含。")


if __name__ == "__main__":
    main()
