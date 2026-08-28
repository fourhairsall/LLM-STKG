"""旅游知识图谱构建 (C1)：LLM 生成/补全 + 时空-语义异构图。

边类型（四类，论文 C2 差异化消息传播的基础）：
  - geo      : 地理邻近（haversine <= geo_radius_km）
  - category : 同类目层级
  - semantic : LLM/文本语义关系（向量化余弦相似度 >= 阈值）
  - covisit  : 用户轨迹共现（共现次数 >= covisit_min）

实现说明：
  - 语义边改用「一次性文本嵌入 + 向量化余弦矩阵」而非逐对 LLM 调用，使真实规模数据可跑；
    真实部署时 text_embedding 可替换为 GPT/GLM 嵌入，语义更准。
  - geo / category / semantic 三类边均用 numpy 向量化构造，避免 Python 双层循环。
  - 大规模生产环境（数万 POI）请用 H3/空间索引加速 geo 边，本脚手架对 num_pois<=~3000 直接矩阵法。
"""
import math
import numpy as np
from .llm_interface import LLMInterface


def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


class TourismKG:
    def __init__(self, cfg, llm: LLMInterface = None):
        self.cfg = cfg
        self.llm = llm or LLMInterface()
        self.edge_index = {}
        self.sem_vecs = None
        self.cat_ids = None
        self.num_cats = None

    def build(self, poi_meta, checkins, sem_vecs=None):
        """poi_meta: list[{poi_id, category, lat, lng, text}]
        checkins: list[(user_id, [poi_id,...])]
        sem_vecs: 可选，预计算语义嵌入（np.ndarray [N, D]）。提供则跳过 LLM 文本编码
                  （C1 真实 BGE 嵌入已离线分块编码并缓存为 poi_bge_emb.npy，避免训练时
                  一次性编码 4980 POI 触发沙箱 segfault）。"""
        n = len(poi_meta)
        lats = np.array([m["lat"] for m in poi_meta], dtype=np.float64)
        lngs = np.array([m["lng"] for m in poi_meta], dtype=np.float64)
        cats = np.array([int(m["category"]) for m in poi_meta], dtype=np.int64)
        texts = [m.get("text", "") for m in poi_meta]
        self.cat_ids = cats.tolist()
        self.num_cats = int(cats.max()) + 1

        if sem_vecs is not None:
            sem = np.asarray(sem_vecs, dtype=np.float64)
        else:
            sem = np.array(self.llm.text_embedding(texts, dim=self.cfg.sem_dim), dtype=np.float64)
        # 归一化用于余弦
        sem_norm = sem / (np.linalg.norm(sem, axis=1, keepdims=True) + 1e-8)
        self.sem_vecs = sem.tolist()

        # ---- geo 边（向量化 haversine 矩阵）----
        # 用等经纬近似距离（小规模 OK）；半径过滤。
        # use_geo_edges=False（数据集无 geo 信息，如 MovieLens/Steam）时整段跳过。
        gi, gj = np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.int64)
        if getattr(self.cfg, "use_geo_edges", True):
            dlat = np.radians(lats[:, None] - lats[None, :])
            dlng = np.radians(lngs[:, None] - lngs[None, :])
            a = (np.sin(dlat / 2) ** 2
                 + np.cos(np.radians(lats[:, None])) * np.cos(np.radians(lats[None, :]))
                 * np.sin(dlng / 2) ** 2)
            dist = 2 * 6371.0 * np.arcsin(np.clip(np.sqrt(a), 0, 1))
            geo_mask = np.triu(dist <= self.cfg.geo_radius_km, k=1)
            gi, gj = np.where(geo_mask)

        # ---- category 边 ----
        # use_category_edges=False（无类目信息的数据集，如 Gowalla/Steam）时跳过。
        ci, cj = np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.int64)
        if getattr(self.cfg, "use_category_edges", True):
            cat_mask = np.triu(cats[:, None] == cats[None, :], k=1)
            ci, cj = np.where(cat_mask)

        # ---- semantic 边（向量化余弦）----
        # use_semantic_edges=False（无 LLM 文本 / 跨域 w/o LLM-text 消融）时跳过。
        si, sj = np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.int64)
        if getattr(self.cfg, "use_semantic_edges", True):
            sim = sem_norm @ sem_norm.T
            sem_mask = np.triu(sim >= self.cfg.semantic_sim_thr, k=1)
            si, sj = np.where(sem_mask)

        # ---- covisit 边 ----
        pair = {}
        freq = {}                     # 单品出现的会话数，PMI / cosine 去热度偏置所需
        n_sess_cov = 0                # 参与统计的会话总数
        for _, seq in checkins:
            s = list(set(int(x) for x in seq))
            if len(s) >= 2:
                n_sess_cov += 1
            for x in s:
                freq[x] = freq.get(x, 0) + 1
            for a in range(len(s)):
                for b in range(a + 1, len(s)):
                    key = (min(s[a], s[b]), max(s[a], s[b]))
                    pair[key] = pair.get(key, 0) + 1
        cov = [(a, b) for (a, b), c in pair.items() if c >= self.cfg.covisit_min]

        # 选边打分：决定每个节点保留哪 k 个共访邻居（covisit_score）。
        # ------------------------------------------------------------------
        # 为什么必须可切换：raw（原始共现次数）在**稠密消费域**会造成枢纽垄断，
        # 进而使 GNN 表征塌缩。实测（hub_collapse_probe.py，每节点 top-10）：
        #
        #   数据集      人均历史   打分     邻居Jaccard  枢纽覆盖  邻居多样性
        #   Steam        76.4     raw       0.5527      0.7755    0.1150
        #                         pmi       0.0295      0.1470    0.8067
        #   MovieLens-1M 234.2    raw       0.3160      0.6065    0.1650
        #                         pmi       0.0179      0.0678    0.7412
        #   Gowalla      20.9     raw       0.0396      0.1853    0.7100  ← 本就健康
        #
        # raw 下 Steam 任意两节点的 top-10 邻居有 55% 重合、前 10 个热门物品独占
        # 77.6% 的边、仅 11.5% 的物品曾被选为邻居 → 均值聚合后各节点消息几乎相同。
        # 对应端到端诊断：POI 两两余弦 0.9997、POI 嵌入梯度 8.75e-05（比 Gowalla 小
        # 470 倍）、C6 中 KG 通道门控被压到 0.0345；关掉图传播（--no_graph）后余弦
        # 立刻回到 0.5618、梯度回到 1.34e-02（↑153×）、R@10 反而从 0.0809 升到 0.0865
        # ——证明塌缩由**图传播本身**引入，且在该域是净负贡献。
        # 因此修复方向不是改池化/改聚合/加深模型，而是给**选边**去热度偏置。
        _cs = str(getattr(self.cfg, "covisit_score", "raw")).lower()

        def _edge_score(a_, b_):
            c_ = pair[(min(a_, b_), max(a_, b_))]
            if _cs == "raw":
                return float(c_)
            fa = max(freq.get(a_, 1), 1)
            fb = max(freq.get(b_, 1), 1)
            if _cs == "cosine":
                return float(c_) / math.sqrt(float(fa) * float(fb))
            if _cs == "pmi":
                # 排序用途，不做 max(·,0) 截断（截断会把负 PMI 全压成并列 0，丢失序）
                return math.log((float(c_) * max(n_sess_cov, 1)) / (float(fa) * float(fb)) + 1e-12)
            raise ValueError(f"未知 covisit_score={_cs}（可选 raw/cosine/pmi）")

        def to_bidir(i, j):
            return np.stack([np.concatenate([i, j]), np.concatenate([j, i])], axis=0)

        # ---- 每类关系的 k-NN 剪枝（max_degree>0 时启用）----
        # 动机：2 km 半径在曼哈顿这类高密度城区会产生平均度数 400+ 的 geo 子图，
        # 均值聚合下等价于对全图取平均 → 过平滑，且单批传播开销上升一个量级。
        # 做法：每个 POI 只保留同类型下「最相关的 k 个」邻居，再做对称化（并集）——
        # 阈值退化为候选集生成器，度数由 k 控制，避免阈值取值直接决定图密度。
        k = int(getattr(self.cfg, "max_degree", 0) or 0)
        if k > 0:
            rng = np.random.default_rng(int(getattr(self.cfg, "seed", 42)))

            def topk_by_score(score, valid, largest=True):
                """每行取 valid 内 score 最优的 k 个，返回对称化后的 [2, E]。"""
                s = np.where(valid, score, (-np.inf if largest else np.inf))
                kk = min(k, n - 1)
                idx = (np.argpartition(-s, kk - 1, axis=1)[:, :kk] if largest
                       else np.argpartition(s, kk - 1, axis=1)[:, :kk])
                rows = np.repeat(np.arange(n), kk)
                cols = idx.reshape(-1)
                ok = np.isfinite(s[rows, cols])
                rows, cols = rows[ok], cols[ok]
                und = np.unique(np.stack([np.minimum(rows, cols),
                                          np.maximum(rows, cols)], axis=1), axis=0)
                if und.size == 0:
                    return np.empty((2, 0), dtype=np.int64)
                return to_bidir(und[:, 0], und[:, 1])

            eye = np.eye(n, dtype=bool)
            geo_e = (topk_by_score(dist, (dist <= self.cfg.geo_radius_km) & ~eye, largest=False)
                     if getattr(self.cfg, "use_geo_edges", True) else np.empty((2, 0), dtype=np.int64))
            sem_e = (topk_by_score(sim, (sim >= self.cfg.semantic_sim_thr) & ~eye, largest=True)
                     if getattr(self.cfg, "use_semantic_edges", True) else np.empty((2, 0), dtype=np.int64))
            # category 边无自然强弱之分：每个 POI 在同类目内均匀抽样 k 个邻居（固定种子）
            if getattr(self.cfg, "use_category_edges", True):
                ci2, cj2 = [], []
                for c in np.unique(cats):
                    members = np.where(cats == c)[0]
                    if members.size < 2:
                        continue
                    for i_ in members:
                        others = members[members != i_]
                        pick = others if others.size <= k else rng.choice(others, size=k, replace=False)
                        ci2.append(np.full(pick.size, i_, dtype=np.int64))
                        cj2.append(np.asarray(pick, dtype=np.int64))
                if ci2:
                    r_, c_ = np.concatenate(ci2), np.concatenate(cj2)
                    und = np.unique(np.stack([np.minimum(r_, c_), np.maximum(r_, c_)], axis=1), axis=0)
                    cat_e = to_bidir(und[:, 0], und[:, 1])
                else:
                    cat_e = np.empty((2, 0), dtype=np.int64)
            else:
                cat_e = np.empty((2, 0), dtype=np.int64)
            # covisit 边：按 covisit_score 选出的相关度保留每 POI 前 k 条邻居。
            # （Foursquare 这类稀疏域平均度数本就 ~3，raw 与 pmi 差异很小；
            #   稠密消费域必须用 pmi/cosine，否则枢纽垄断 → 表征塌缩，见上方注释。）
            # use_covisit_edges=False（P1-5 单边缘类型消融：covisit-only 对照）时整段跳过。
            if cov and getattr(self.cfg, "use_covisit_edges", True):
                bycnt = {}
                for (a_, b_) in cov:
                    c_ = _edge_score(a_, b_)
                    bycnt.setdefault(a_, []).append((c_, b_))
                    bycnt.setdefault(b_, []).append((c_, a_))
                keep = set()
                for i_, lst in bycnt.items():
                    for _, j_ in sorted(lst, key=lambda x: -x[0])[:k]:
                        keep.add((min(i_, j_), max(i_, j_)))
                arr = np.array(sorted(keep), dtype=np.int64)
                cov_e = to_bidir(arr[:, 0], arr[:, 1])
            else:
                cov_e = np.empty((2, 0), dtype=np.int64)
            self.edge_index = {"geo": geo_e, "category": cat_e,
                               "semantic": sem_e, "covisit": cov_e}
            return self

        self.edge_index = {
            "geo": to_bidir(gi, gj),
            "category": to_bidir(ci, cj),
            "semantic": to_bidir(si, sj),
            "covisit": (np.array(cov, dtype=np.int64).T if (cov and getattr(self.cfg, "use_covisit_edges", True))
                        else np.empty((2, 0), dtype=np.int64)),
        }
        return self

    def stats(self):
        return {k: (v.shape[1] if hasattr(v, "shape") and v.ndim == 2 else len(v)) // 2
                for k, v in self.edge_index.items()}
