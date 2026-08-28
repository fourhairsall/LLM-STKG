"""平凡（trivial / non-learned）基线：审稿人第一时间会问的"不学习也能做到多少"。

背景（为什么必须有这一组）
--------------------------------
Foursquare-NYC 的 LLM4POI 划分中，**75.7% 的测试目标已经出现在该样本自己的历史序列里**
（target ∈ history）。在这种"重访主导"的评测协议下，一个完全不学习的规则——
"预测历史里出现次数最多的 POI"——就能取得很高的 Recall。若论文只报全量 Recall 而不报
这类基线，读者无法判断模型学到的究竟是"时空-语义规律"还是"复读历史"。

因此本模块提供三个零参数 / 极少参数的基线，全部在同一全候选排名协议下评估：

  * History-Frequency (HF)  : s(p) = 该 POI 在本样本历史中出现的次数
  * History-Recency  (HR)   : s(p) = 1 / (该 POI 最近一次出现距序列末尾的步数)
  * Markov-1         (MC1)  : s(p) = 训练集中 last_poi -> p 的一阶转移计数

三者都用「训练集热度」做极小权重的兜底（backoff），只用于给未被规则覆盖的 POI
一个确定性的排序，不改变规则本身的相对顺序（backoff 权重 << 规则最小间隔）。
没有 backoff 时大量 POI 同分，argsort 的顺序由实现细节决定，指标不可复现。

用法
----
    from .trivial_baselines import build_trivial_scores
    mats = build_trivial_scores(test_samples, num_pois, train_checkins)
    for name, S in mats.items():
        metrics = rank_metrics(S, targets)
"""
from collections import Counter, defaultdict

import torch


# backoff 权重：远小于任何规则分数的最小间隔（规则分数最小间隔为 1 或 1/len(hist)），
# 保证兜底只在"规则同分"时起作用，不会把规则外的 POI 抬到规则内 POI 之上。
_BACKOFF_W = 1e-6


def _pop_vector(checkins, num_pois):
    """训练集 POI 频次，归一化到 [0, 1]，仅用作 backoff。"""
    freq = Counter(p for _, seq in checkins for p in seq)
    v = torch.zeros(num_pois, dtype=torch.float32)
    for p, c in freq.items():
        if 0 <= p < num_pois:
            v[p] = float(c)
    if v.max() > 0:
        v = v / v.max()
    return v, freq


def _transition_table(checkins, num_pois):
    """一阶转移计数 T[a][b] = 训练集中 a 紧接着出现 b 的次数。用 dict-of-Counter 省内存。"""
    T = defaultdict(Counter)
    for _, seq in checkins:
        for a, b in zip(seq[:-1], seq[1:]):
            if 0 <= a < num_pois and 0 <= b < num_pois:
                T[a][b] += 1
    return T


def build_trivial_scores(test_samples, num_pois, checkins):
    """返回 {baseline_name: [B, num_pois] float32 打分矩阵}。

    test_samples : list[(uid, history_list, target)]
    checkins     : 训练集 [(uid, seq)]，仅用于 popularity backoff 与一阶转移表
    """
    B = len(test_samples)
    pop, _ = _pop_vector(checkins, num_pois)
    T = _transition_table(checkins, num_pois)

    S_hf = pop.unsqueeze(0).repeat(B, 1) * _BACKOFF_W
    S_hr = S_hf.clone()
    S_mc = S_hf.clone()

    for i, (_, hist, _) in enumerate(test_samples):
        h = [int(p) for p in hist if 0 <= int(p) < num_pois]
        if not h:
            continue
        # HF：历史出现次数
        for p, c in Counter(h).items():
            S_hf[i, p] += float(c)
        # HR：最近一次出现距末尾的步数（末尾为 1 步）
        L = len(h)
        for j, p in enumerate(h):
            S_hr[i, p] = max(float(S_hr[i, p]), 1.0 / float(L - j))
        # MC1：从最后一个 POI 出发的一阶转移计数
        last = h[-1]
        for p, c in T.get(last, {}).items():
            S_mc[i, p] += float(c)

    return {
        "History-Freq (HF)": S_hf,
        "History-Recency (HR)": S_hr,
        "Markov-1 (MC1)": S_mc,
    }


def split_test_subsets(test_samples, checkins, cold_thr=5):
    """按「重访 / 新颖」与「冷启动」二维切分测试集，返回 {subset_name: index_list}。

    - revisit : target ∈ history（模型只需从历史里挑对一个）
    - novel   : target ∉ history（真正的"预测没去过的地方"）
    - cold    : target 训练频次 ≤ cold_thr
    """
    freq = Counter(p for _, seq in checkins for p in seq)
    idx = {"all": [], "revisit": [], "novel": [], "cold": [],
           "novel_cold": [], "revisit_cold": []}
    for i, (_, hist, tgt) in enumerate(test_samples):
        hs = set(int(p) for p in hist)
        is_rev = int(tgt) in hs
        is_cold = freq.get(int(tgt), 0) <= cold_thr
        idx["all"].append(i)
        idx["revisit" if is_rev else "novel"].append(i)
        if is_cold:
            idx["cold"].append(i)
            idx["revisit_cold" if is_rev else "novel_cold"].append(i)
    return idx
