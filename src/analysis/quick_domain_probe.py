"""跨域数据集"难度水位"探针（纯 CPU，无需训练）。

目的：在把 GPU 预算投进完整基线之前，先用三个零参数规则确定一份切分的合理水位，
用来判断"模型分数低"到底是**数据本身难**还是**模型没学到东西**：

  1. Random      —— 理论下界（k / 候选数）。
  2. Popularity  —— 只按训练集频次排序，完全不看历史。
  3. ItemKNN-cooc—— 经典共现近邻：score(j) = Σ_{i∈hist} cooc[i][j]（已行归一化），
                    并屏蔽历史中已出现的物品（无重复消费域）。这是"结构信号"的强规则上界，
                    我们的 C6-cooc 通道若显著低于它，说明模型没把共现信号用起来。

用法：
  python quick_domain_probe.py --proc ../data/steam/processed
  python quick_domain_probe.py --proc ../data/ml-1m/processed --mask_hist
"""
import argparse
import json
import os

import numpy as np


def recall_ndcg(ranks, k_list=(5, 10)):
    ranks = np.asarray(ranks, dtype=np.float64)
    out = {}
    for k in k_list:
        hit = (ranks <= k)
        out[f"Recall@{k}"] = float(hit.mean())
        out[f"NDCG@{k}"] = float((hit / np.log2(ranks + 1)).mean())
    out["median_rank"] = float(np.median(ranks))
    out["MRR"] = float((1.0 / ranks).mean())
    return out


def rank_of(scores, target, mask_idx=None):
    """target 在 scores 降序中的 1-based 排名；mask_idx 内的物品置 -inf。

    ⚠️ 历史修复：此前 mask_idx 由调用方按 `[h for h in hist if h != tgt]` 构造，
    即"屏蔽历史但把目标摘出去"——那是**标签泄漏**（删掉竞争者却保留正确答案），
    会在重访域把指标虚高（实测 Gowalla ItemKNN R@10 从 0.2161 假涨到 0.2721）。
    现改为无条件屏蔽整个历史：若目标本身就在历史里，它同样被置 -inf，
    该样本自然无法命中——这正是"无重复消费域协议"的应有代价，也正因如此
    该协议**只能用于 revisit_ratio≈0 的域**（Steam/MovieLens），
    重访主导域（Gowalla 0.6759 / Foursquare 0.7574）必须关闭。
    """
    s = scores.copy()
    if mask_idx is not None and len(mask_idx):
        s[mask_idx] = -np.inf
    t = s[target]
    if not np.isfinite(t):        # 目标被屏蔽 → 视为排到最末，不得命中
        return len(s)
    # 【并列口径必须与主评测一致】evaluate.target_rank 默认 tie="pessimistic"：
    #   rank = 1 + #{s > t} + #{s == t, 非自身}
    # 此前本探针用乐观口径 1 + #{s > t}，会把同分候选**全部算作并列第一**。
    # 热度 / 共现这类零参数规则同分极多（Steam 上 cooc-max 聚合下 50.8% 候选并列榜首），
    # 乐观口径会把它们的 Recall 系统性抬高，再拿去和悲观口径的模型比就是虚假对照。
    return int(1 + np.sum(s > t) + max(0, np.sum(s == t) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", required=True, help="processed 目录")
    ap.add_argument("--mask_hist", action="store_true",
                    help="打分时屏蔽历史中已出现的物品（无重复消费域应开启）")
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 个测试样本（0=全部）")
    args = ap.parse_args()

    proc = args.proc
    stats = json.load(open(os.path.join(proc, "stats.json"), encoding="utf-8"))
    train = json.load(open(os.path.join(proc, "train_checkins.json"), encoding="utf-8"))
    test = json.load(open(os.path.join(proc, "test_samples.json"), encoding="utf-8"))
    cooc = np.load(os.path.join(proc, "cooc_matrix.npy")).astype(np.float32)
    N = cooc.shape[0]
    print(f"[probe] {stats.get('dataset')} N={N} train_seq={len(train)} test={len(test)}")

    # 训练频次 → Popularity
    pop = np.zeros(N, dtype=np.float32)
    for rec in train:
        seq = (rec[1] if isinstance(rec, (list, tuple))
               else rec.get("pois", rec.get("sequence", [])))
        for p in seq:
            if 0 <= p < N:
                pop[p] += 1.0

    samples = test[: args.limit] if args.limit else test
    # test_samples.json 结构兼容：[uid, hist, target] 或 {"history":..,"target":..}
    def unpack(s):
        if isinstance(s, dict):
            return s.get("history", s.get("hist")), s["target"]
        if len(s) == 3:
            return s[1], s[2]
        return s[0], s[1]

    rv = 0
    ranks_pop, ranks_knn, ranks_rand = [], [], []
    rng = np.random.default_rng(42)
    for s in samples:
        hist, tgt = unpack(s)
        hist = [h for h in hist if 0 <= h < N]
        if not (0 <= tgt < N):
            continue
        if tgt in set(hist):
            rv += 1
        # 无条件屏蔽整个历史（含恰好等于目标的情形）——见 rank_of 的说明，
        # 摘出目标即标签泄漏。故本开关仅可用于 revisit_ratio≈0 的域。
        mask = np.array(sorted(set(hist)), dtype=np.int64) if (args.mask_hist and hist) else None
        ranks_pop.append(rank_of(pop, tgt, mask))
        knn = cooc[hist].sum(axis=0) if hist else np.zeros(N, dtype=np.float32)
        ranks_knn.append(rank_of(knn, tgt, mask))
        ranks_rand.append(rank_of(rng.random(N).astype(np.float32), tgt, mask))

    n = len(ranks_pop)
    print(f"[probe] n_eval={n} revisit_ratio={rv / max(n,1):.4f} mask_hist={args.mask_hist}")
    for name, r in (("Random", ranks_rand), ("Popularity", ranks_pop), ("ItemKNN-cooc", ranks_knn)):
        m = recall_ndcg(r)
        print(f"  {name:<14} R@5={m['Recall@5']:.4f} R@10={m['Recall@10']:.4f} "
              f"N@10={m['NDCG@10']:.4f} median_rank={m['median_rank']:.0f}")


if __name__ == "__main__":
    main()
