"""评估指标：Recall@k / NDCG@k（与 SOTA 对比的标准指标）。"""
import torch


def metrics_from_ranks(rank: torch.Tensor, k_list=(5, 10)):
    """由 1-based 排名张量直接算指标。单相关项（每样本只有 1 个正例）协议。

    ⚠️ 修正记录：此前 NDCG 被实现为 ``1/rank``（截断 MRR），并非 NDCG。
    单相关项时 IDCG = 1，DCG = 1/log2(rank+1)，故 NDCG@k = 1/log2(rank+1) if rank<=k else 0。
    旧公式在 rank=2 给 0.500（正确值 0.6309）、rank=3 给 0.333（正确值 0.500），
    系统性低估且与文献中按标准 NDCG 报告的 SOTA 数值不可比。现同时给出：
      - ``NDCG@k`` : 标准定义 1/log2(rank+1)
      - ``MRR@k``  : 截断倒数排名 1/rank（即旧代码所报的量，保留以便与历史结果对照）
    """
    rank = rank.float()
    res = {}
    for k in k_list:
        in_top = rank <= k
        res[f"Recall@{k}"] = round(in_top.float().mean().item(), 4)
        ndcg = torch.zeros_like(rank)
        ndcg[in_top] = 1.0 / torch.log2(rank[in_top] + 1.0)
        res[f"NDCG@{k}"] = round(ndcg.mean().item(), 4)
        mrr = torch.zeros_like(rank)
        mrr[in_top] = 1.0 / rank[in_top]
        res[f"MRR@{k}"] = round(mrr.mean().item(), 4)
    res["MRR"] = round((1.0 / rank).mean().item(), 4)
    return res


def target_rank(scores: torch.Tensor, targets: torch.Tensor, tie: str = "pessimistic"):
    """返回目标的 1-based 排名 [B]，显式处理同分。

    tie='pessimistic'：并列时取最差名次（rank = 1 + #{s > s_t} + #{s == s_t, 非自身}）。
    非学习型基线（热度 / 历史规则）会产生大量同分，argsort 的隐式并列顺序由列索引决定，
    会给低索引 POI 无根据的优势；用悲观并列可避免高估平凡基线，也让结果与实现无关。
    """
    s_t = scores.gather(1, targets.unsqueeze(1))           # [B,1]
    greater = (scores > s_t).sum(dim=1)
    equal = (scores == s_t).sum(dim=1) - 1                  # 去掉目标自身
    if tie == "optimistic":
        return (greater + 1).long()
    return (greater + equal + 1).long()


def mask_history(scores: torch.Tensor, samples, num_pois: int):
    """把每个样本历史中已出现过的物品分数置 -inf（**无条件**，包含恰好等于目标的情形）。

    为什么需要，以及为什么只能在"无重复消费域"开启
    ------------------------------------------------
    MovieLens / Steam 这类数据集不存在重复消费（测试端重访率≈0），SASRec 一系工作的标准
    评测协议本来就把用户已交互过的物品排除在候选之外。若不排除，模型会把几十个历史物品顶到
    榜首、白白占满 top-10，Recall 被系统性压低——实测 Steam 上仅"是否屏蔽历史"一项就让
    零参数 Popularity 的 R@10 从 0.0448 变成 0.0725（1.6 倍差距）。

    ⚠️ 必须**无条件**屏蔽，不能写成"除目标外都屏蔽"：那等于删掉该样本的竞争者却保留正确
    答案，是标签泄漏。因此在 Foursquare-NYC 这类重访主导域（测试端重访率 0.757）**绝不能
    开启**，否则多数目标会被自身历史屏蔽。开关由数据集 revisit_ratio 决定，且必须对 ours
    与全部基线**同时、同样**施加，否则比较不公平。

    scores : [B, N]，就地修改并返回
    samples: 与 scores 行一一对应的 (uid, history, target) 序列
    """
    if not len(samples):
        return scores
    rows, cols = [], []
    for i, s in enumerate(samples):
        for p in set(s[1]):
            if 0 <= p < num_pois:
                rows.append(i)
                cols.append(p)
    if rows:
        scores[torch.tensor(rows, dtype=torch.long),
               torch.tensor(cols, dtype=torch.long)] = float("-inf")
    return scores


def rank_metrics(scores: torch.Tensor, targets: torch.Tensor, k_list=(5, 10)):
    """scores: [B, N] 全候选打分；targets: [B] 真实下一 POI 的 id（即列索引）。"""
    return metrics_from_ranks(target_rank(scores, targets).float(), k_list=k_list)


def rank_diag(scores: torch.Tensor, targets: torch.Tensor):
    """NDCG 诊断：正样本在【全候选】排名中的分布 + top-K 分数离散度。
    直接回答「模型是否把正样本排到第 1 名」——若 pct_rank1 低而 median_rank 高，
    则 R@K 仍高（正样本在 top-K）但 NDCG 低（正样本常在 rank 5~10）。
    """
    B = scores.size(0)
    rank = target_rank(scores, targets).float()   # 1-based，悲观并列（与 rank_metrics 一致）
    topk = torch.topk(scores, 10, dim=1).values
    bins = [(1, 1, "rank1"), (2, 3, "rank2-3"), (4, 5, "rank4-5"),
            (6, 10, "rank6-10"), (11, 50, "rank11-50"),
            (51, 200, "rank51-200"), (201, 10**9, "rank200+")]
    hist = {}
    for lo, hi, name in bins:
        hist[name] = float(((rank >= lo) & (rank <= hi)).float().mean().item())
    # top-10 分数离散度：越高=top 候选分数越有区分度（排序更"自信"）
    top10_std = float(topk.std(dim=1).mean().item())
    # 正样本分数 / 榜首分数：越接近 1=正样本越靠近榜首
    tgt_score = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
    top1_score = topk[:, 0]
    ratio = float((tgt_score / (top1_score + 1e-8)).mean().item())
    # 分数分布校准证据（CE vs BPR/listwise）：目标分数相对全候选分布的标准化位置。
    # z_target = (s_target - mean(s)) / std(s)，越大表示目标在整个候选分布中越突出；
    # 若某训练目标只在少量负样本内可分（塌缩），z 会显著变小、而 top10_score_std 也随之退化。
    mu = scores.mean(dim=1)
    sd = scores.std(dim=1) + 1e-8
    z_tgt = ((tgt_score - mu) / sd)
    z_top1 = ((top1_score - mu) / sd)
    return {
        "pct_rank1": round(float((rank == 1).float().mean().item()), 4),
        "median_rank": round(float(rank.median().item()), 2),
        "rank_hist": {k: round(v, 4) for k, v in hist.items()},
        "top10_score_std": round(top10_std, 4),
        "target_vs_top1_ratio": round(ratio, 4),
        "z_target_mean": round(float(z_tgt.mean().item()), 4),
        "z_top1_mean": round(float(z_top1.mean().item()), 4),
        "score_std_mean": round(float(sd.mean().item()), 4),
        # 逐样本 1-based 排名：供离线做配对显著性检验（McNemar / 配对 bootstrap / Wilcoxon）
        "ranks": [int(r) for r in rank.tolist()],
    }
