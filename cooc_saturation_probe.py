"""C6-cooc 通道聚合方式的饱和度量化探针（CPU，只读 processed 目录）。

动机
----
跨域实测发现：Steam 上 LLM-STKG 精确退化到热度水平（R@10=0.0729 vs Popularity 0.0725），
且换序列编码器（mean→GRU）无任何改善（0.0725）。由于跨域配置下 C6 只剩 `pop,cooc`
两个通道，若 cooc 通道不含判别信息，模型就只能靠 pop —— 这与实测完全吻合。

本脚本直接量化 cooc 通道在两种聚合方式下的判别力：
  max : feat(c) = max_{h∈hist ∩ topk(c)} cooc[c, h]      （原实现）
  sum : feat(c) = log1p( Σ_{h∈hist ∩ topk(c)} cooc[c, h] )（长历史域新实现）

报告三项指标（越高越糟的用 ↓ 标注）：
  · sat_ratio ↓ : 特征值 > 0.9 的候选占比（饱和到上界、彼此不可区分）
  · tie_ratio ↓ : 落在**最大值**附近（>0.99×max）的候选占比 —— 直接决定 top-K 是否被并列淹没
  · uniq_ratio  : 去重后不同特征值的个数 / 候选数 —— 判别粒度
另给出该聚合方式单独作为打分器的 Recall@10（不训练，纯规则），
与 ItemKNN（Σ_{h∈hist} cooc[h,c]，全量无 top-k 截断）对照。

用法：
  python cooc_saturation_probe.py --proc ../data/steam/processed --mask_hist
  python cooc_saturation_probe.py --proc ../data/gowalla/processed
"""
import argparse
import json
import os

import numpy as np


def topk_sparse(cooc, k):
    """每行保留 top-k 邻居（与 model 侧 cooc_topk 稀疏化一致）。返回 idx[N,k], val[N,k]。"""
    N = cooc.shape[0]
    k = min(k, N)
    idx = np.argpartition(-cooc, kth=k - 1, axis=1)[:, :k]
    val = np.take_along_axis(cooc, idx, axis=1)
    return idx.astype(np.int64), val.astype(np.float32)


def rank_of(scores, target, mask_idx):
    """悲观并列口径，与 evaluate.target_rank(tie="pessimistic") 一致。

    用乐观口径（1+#{s>t}）会把同分候选全算作并列第一 —— 在本脚本考察的
    max 聚合下 50% 候选同分，乐观口径会给出 R@10≈0.46 的荒谬数字。
    """
    s = scores.copy()
    if mask_idx is not None and len(mask_idx):
        s[mask_idx] = -np.inf
    t = s[target]
    if not np.isfinite(t):
        return len(s)
    return int(1 + np.sum(s > t) + max(0, np.sum(s == t) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", required=True)
    ap.add_argument("--topk", type=int, default=50, help="与 cfg.cooc_topk 一致")
    ap.add_argument("--mask_hist", action="store_true")
    ap.add_argument("--limit", type=int, default=600, help="评测样本数（0=全部）")
    args = ap.parse_args()

    stats = json.load(open(os.path.join(args.proc, "stats.json"), encoding="utf-8"))
    test = json.load(open(os.path.join(args.proc, "test_samples.json"), encoding="utf-8"))
    cooc = np.load(os.path.join(args.proc, "cooc_matrix.npy")).astype(np.float32)
    N = cooc.shape[0]
    idx, val = topk_sparse(cooc, args.topk)
    samples = test[: args.limit] if args.limit else test
    print(f"[probe] {stats.get('dataset')} N={N} topk={args.topk} "
          f"n_eval={len(samples)} mask_hist={args.mask_hist}")

    acc = {m: {"sat": [], "tie": [], "uniq": [], "ranks": []} for m in ("max", "sum")}
    ranks_knn = []
    hist_lens = []
    for rec in samples:
        hist = rec["history"] if isinstance(rec, dict) else rec[1]
        tgt = rec["target"] if isinstance(rec, dict) else rec[2]
        hist = [h for h in hist if 0 <= h < N]
        if not hist:
            continue
        hist_lens.append(len(hist))
        in_hist = np.zeros(N, dtype=np.float32)
        in_hist[np.array(hist, dtype=np.int64)] = 1.0
        hit = val * in_hist[idx]                       # [N, topk] 命中历史的共现强度
        feats = {"max": hit.max(axis=1),
                 "sum": np.log1p(hit.sum(axis=1))}
        mask = np.array(sorted(set(hist)), dtype=np.int64) if (args.mask_hist and hist) else None
        for m, f in feats.items():
            hi = f.max()
            acc[m]["sat"].append(float((f > 0.9).mean()))
            acc[m]["tie"].append(float((f >= 0.99 * hi).mean()) if hi > 0 else 1.0)
            acc[m]["uniq"].append(len(np.unique(np.round(f, 6))) / N)
            acc[m]["ranks"].append(rank_of(f, tgt, mask))
        ranks_knn.append(rank_of(cooc[np.array(hist, dtype=np.int64)].sum(axis=0), tgt, mask))

    n = len(ranks_knn)
    print(f"[probe] 有效样本={n} 历史均长={np.mean(hist_lens):.1f}")
    print(f"{'聚合':<6}{'sat_ratio↓':>12}{'tie_ratio↓':>12}{'uniq_ratio':>12}{'Recall@10':>12}")
    for m in ("max", "sum"):
        a = acc[m]
        r = np.array(a["ranks"])
        print(f"{m:<6}{np.mean(a['sat']):>12.4f}{np.mean(a['tie']):>12.4f}"
              f"{np.mean(a['uniq']):>12.4f}{float((r <= 10).mean()):>12.4f}")
    rk = np.array(ranks_knn)
    print(f"{'ItemKNN(全量Σ)':<6}{'—':>18}{'—':>12}{'—':>10}{float((rk <= 10).mean()):>12.4f}")
    print("\n判读：tie_ratio 越高说明越多候选并列在最高分附近，top-K 由并列噪声决定；"
          "若 max 的 tie_ratio 远高于 sum，则 max 聚合是该域 cooc 通道失效的直接原因。")


if __name__ == "__main__":
    main()
