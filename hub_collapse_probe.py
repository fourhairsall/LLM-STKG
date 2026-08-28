# -*- coding: utf-8 -*-
"""共访图「枢纽塌缩」探针：解释 Steam 上 GNN 传播为何导致 POI 表征塌缩。

背景（实测证据链）
------------------
Steam(600 POI) 上开图传播 vs 关图传播的训练诊断对照：

    指标                    有图传播      --no_graph      变化
    POI 两两余弦            0.99971       0.5618          塌缩消失
    POI 嵌入梯度范数        8.75e-05      1.341e-02       ↑153×
    KG 通道门控             0.0345        0.112           ↑3.2×
    Recall@10               0.0809        0.0865          图传播是净负贡献

即：塌缩由**图传播本身**引入，而非节点特征贫瘠的被动结果。本探针检验塌缩的
具体机制假设：

    H：kg_builder 的 covisit 边按**原始共现次数** pair[(a,b)] 取每节点 top-k。
       在稠密消费域（Steam 人均历史 74.5、候选仅 600），任意物品与头部热门物品的
       原始共现次数都很大，于是**几乎每个节点的 top-k 邻居都是同一批枢纽物品**。
       均值聚合下所有节点收到近乎相同的消息 → 表征塌成一根 → 点积打分无区分度。

若 H 成立，则修复方向不是"改池化/改聚合/加深模型"，而是**给选边打分去热度偏置**
（如用 PMI / cosine 归一化替代原始计数），使邻居集合恢复多样性。

度量
----
1. jaccard_mean : 随机节点对的 top-k 邻居集合 Jaccard 相似度均值（越高越塌缩）
2. hub_cover    : 被最多节点选为邻居的前 10 个物品，覆盖了全部有向边的比例
3. indeg_gini   : 「被选为邻居」次数分布的 Gini 系数（1=完全集中于枢纽）
4. uniq_ratio   : 出现在任意 top-k 邻居集合中的不同物品数 / 总物品数

对三种选边打分各算一遍：
  raw    : pair[(a,b)]                          （现状）
  pmi    : log( p(a,b) / (p(a)p(b)) ) 正部       （去热度偏置）
  cosine : cooc(a,b)/sqrt(cnt_a*cnt_b)          （去热度偏置，更平滑）

用法：python hub_collapse_probe.py            （跑全部可用数据集）
      python hub_collapse_probe.py steam      （只跑指定数据集）
纯 CPU、只读 processed 目录，不动 GPU、不改任何训练产物。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# 仓库里存在两个 data/ 目录：旅游推荐论文/data（通用加载器产物）与上一级 2026年7月/data
# （real_data_prepare.py 的 Foursquare 产物）。两处都要找。
_ROOTS = [os.path.join(_HERE, "..", "data"), os.path.join(_HERE, "..", "..", "data")]


def _find(*parts):
    """在候选 data 根目录中返回第一个存在的路径；都不存在则返回第一个（用于报错提示）。"""
    for r in _ROOTS:
        p = os.path.join(r, *parts)
        if os.path.isdir(p) and os.listdir(p):
            return p
    return os.path.join(_ROOTS[0], *parts)


ROOT = _ROOTS[0]
DATASETS = {
    "steam": _find("steam", "processed"),
    "gowalla": _find("gowalla", "processed"),
    "movielens-1m": _find("ml-1m", "processed"),
    # Foursquare-NYC 走 real_data_prepare.py 的旧格式（train_trajs.json / poi_meta.json），
    # 与通用加载器的 processed 布局不同，故在 _load 里单独兼容。它是本文主数据集，
    # 必须纳入对照才能说明"塌缩只发生在稠密消费域"。
    "foursquare": _find("real_foursquare_nyc", "processed"),
}
TOPK = 10          # 与 config.max_degree 默认一致


def _gini(x: np.ndarray) -> float:
    """Gini 系数：0=完全均匀，1=完全集中。"""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum()) / (n * x.sum()) - (n + 1.0) / n)


def _seq_of(rec):
    """兼容 processed 的两种记录形态：{"user_id":u,"pois":[...]} 或 (u, [...]) / [...]。"""
    if isinstance(rec, dict):
        return rec.get("pois") or rec.get("seq") or []
    if isinstance(rec, (list, tuple)) and len(rec) >= 2 and not isinstance(rec[0], (list, tuple)):
        return rec[1]
    return rec


def _pair_counts(checkins, n_poi: int):
    """由训练轨迹统计 [N,N] 共现计数（对称，去重后按会话内两两计），及单品频次。"""
    cnt = np.zeros(n_poi, dtype=np.float64)
    co = np.zeros((n_poi, n_poi), dtype=np.float64)
    n_sess = 0
    for rec in checkins:
        seq = _seq_of(rec)
        s = sorted({int(x) for x in seq if 0 <= int(x) < n_poi})
        if len(s) < 2:
            continue
        n_sess += 1
        idx = np.array(s, dtype=np.int64)
        cnt[idx] += 1.0
        co[np.ix_(idx, idx)] += 1.0
    np.fill_diagonal(co, 0.0)
    return co, cnt, n_sess


def _score_matrix(co: np.ndarray, cnt: np.ndarray, n_sess: int, mode: str) -> np.ndarray:
    """三种选边打分。注意都必须对称，否则 top-k 后对称化会引入额外偏置。"""
    if mode == "raw":
        return co
    if mode == "cosine":
        d = np.sqrt(np.outer(cnt, cnt)) + 1e-9
        return co / d
    if mode == "pmi":
        # p(a,b)=co/n_sess, p(a)=cnt/n_sess → pmi = log(co*n_sess/(cnt_a*cnt_b))
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.log((co * max(n_sess, 1)) / (np.outer(cnt, cnt) + 1e-9) + 1e-12)
        return np.maximum(m, 0.0) * (co > 0)
    raise ValueError(mode)


def _topk_neighbors(score: np.ndarray, k: int):
    """每行取分值最大的 k 个（排除自身、排除 0 分），返回 list[set]。"""
    n = score.shape[0]
    out = []
    kk = min(k, n - 1)
    for i in range(n):
        row = score[i].copy()
        row[i] = -np.inf
        if kk <= 0:
            out.append(set())
            continue
        cand = np.argpartition(-row, kk - 1)[:kk]
        cand = cand[row[cand] > 0]          # 0 分（无共现）不成边
        out.append(set(int(j) for j in cand))
    return out


def _metrics(nbrs, n_poi: int, rng, n_pairs: int = 20000):
    non_empty = [i for i, s in enumerate(nbrs) if s]
    if len(non_empty) < 2:
        return dict(jaccard_mean=0.0, hub_cover=0.0, indeg_gini=0.0,
                    uniq_ratio=0.0, mean_deg=0.0, n_active=len(non_empty))
    # 1) 随机节点对邻居集合 Jaccard
    a = rng.choice(non_empty, size=n_pairs)
    b = rng.choice(non_empty, size=n_pairs)
    js = []
    for i, j in zip(a, b):
        if i == j:
            continue
        si, sj = nbrs[i], nbrs[j]
        u = len(si | sj)
        js.append(len(si & sj) / u if u else 0.0)
    # 2) 入选次数分布
    indeg = np.zeros(n_poi, dtype=np.float64)
    total_edges = 0
    for s in nbrs:
        total_edges += len(s)
        for j in s:
            indeg[j] += 1.0
    top10 = np.sort(indeg)[::-1][:10].sum()
    return dict(
        jaccard_mean=float(np.mean(js)) if js else 0.0,
        hub_cover=float(top10 / max(total_edges, 1)),
        indeg_gini=_gini(indeg),
        uniq_ratio=float((indeg > 0).sum() / max(n_poi, 1)),
        mean_deg=float(total_edges / max(len(non_empty), 1)),
        n_active=len(non_empty),
    )


def _load(path: str):
    """返回 (checkins, n_poi)。兼容通用加载器布局与 Foursquare 旧布局。"""
    f_ck = os.path.join(path, "train_checkins.json")
    f_poi = os.path.join(path, "pois.json")
    if os.path.exists(f_ck) and os.path.exists(f_poi):
        with open(f_ck, encoding="utf-8") as fh:
            checkins = json.load(fh)
        with open(f_poi, encoding="utf-8") as fh:
            pois = json.load(fh)
        return checkins, len(pois)
    f_tr = os.path.join(path, "train_trajs.json")
    f_meta = os.path.join(path, "poi_meta.json")
    if os.path.exists(f_tr) and os.path.exists(f_meta):
        with open(f_tr, encoding="utf-8") as fh:
            trajs = json.load(fh)
        with open(f_meta, encoding="utf-8") as fh:
            meta = json.load(fh)
        return trajs, len(meta)
    return None, 0


def run(name: str, path: str, topk: int = TOPK):
    checkins, n_poi = _load(path)
    if not checkins or n_poi <= 0:
        print(f"[skip] {name}: 缺 processed 文件 ({path})")
        return None
    co, cnt, n_sess = _pair_counts(checkins, n_poi)
    hist_len = np.array([len({int(x) for x in _seq_of(r)}) for r in checkins],
                        dtype=np.float64)
    rng = np.random.default_rng(42)
    print(f"\n=== {name} | N_poi={n_poi} 训练会话={n_sess} 人均去重历史={hist_len.mean():.1f} "
          f"图密度上界={n_poi and float((co > 0).sum()) / max(n_poi * (n_poi - 1), 1):.3f} ===")
    print(f"{'选边打分':<8}{'Jaccard↓':>10}{'枢纽覆盖↓':>11}{'入度Gini↓':>11}"
          f"{'邻居多样性↑':>13}{'平均度':>8}")
    res = {}
    for mode in ("raw", "cosine", "pmi"):
        sc = _score_matrix(co, cnt, n_sess, mode)
        nb = _topk_neighbors(sc, topk)
        m = _metrics(nb, n_poi, rng)
        res[mode] = m
        print(f"{mode:<8}{m['jaccard_mean']:>10.4f}{m['hub_cover']:>11.4f}"
              f"{m['indeg_gini']:>11.4f}{m['uniq_ratio']:>13.4f}{m['mean_deg']:>8.1f}")
    return {"dataset": name, "n_poi": n_poi, "n_sessions": n_sess,
            "hist_len_mean": round(float(hist_len.mean()), 2), "topk": topk,
            "metrics": res}


def main():
    keys = sys.argv[1:] or list(DATASETS)
    out = []
    for k in keys:
        if k not in DATASETS:
            print(f"[skip] 未知数据集 {k}")
            continue
        r = run(k, DATASETS[k])
        if r:
            out.append(r)
    if out:
        dst = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "hub_collapse_probe.json")
        with open(dst, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"\n已保存 -> {dst}")
        print("\n判读：Jaccard / 枢纽覆盖 / 入度Gini 越高 = 邻居集合越被同一批热门物品垄断，"
              "\n      均值聚合后各节点消息越相同 → GNN 表征塌缩越严重。"
              "\n      若 raw 在 Steam 上显著劣于 cosine/pmi，则修复方向为选边去热度偏置。")


if __name__ == "__main__":
    main()
