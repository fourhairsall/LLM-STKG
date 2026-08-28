"""KG 路径解释模块 (C3 可解释性的**可执行实现**) —— 论文 §4.5 / §5.8 的支撑代码。

背景与动机
----------
此前稿件在 §4.5 / §5.8 描述了"沿知识图谱路径生成推荐解释"，但代码库中并无对应实现，
案例也非取自真实数据。本模块补上真实实现，并且**只报告可从数据算出的量**：
不做"解释质量"的主观宣称，改为给出三类可核查的代理证据。

方法
----
把 C1/C2 构建的四类边（geo / category / semantic / covisit）视为一张无向多关系图 G。
给定一个会话历史 H = {p_1..p_m} 与被推荐的 POI q：

  1) 多源最短路：在 G 上以 H 为源集做 BFS，得到 d(H, q) = min_i dist(p_i, q)（跳数）。
  2) 路径抽取：回溯得到一条最短路径及其逐跳关系类型，形成人类可读解释链，例如
        Coffee Shop near 40.73,-73.99  --semantic-->  Bakery near 40.73,-74.00
                                       --covisit-->   Park near 40.73,-74.00
  3) 关系归因：统计路径首跳关系类型分布，回答"解释主要由哪种关系承载"。

代理指标（均为可核查的客观量，不含主观评分）
--------------------------------------------
  - reach@L        : top-1 推荐在 ≤L 跳内可从历史到达的比例（L=1,2,3）
  - 随机/热度对照  : 同一指标在「随机 POI」与「Popularity top-1」上的取值。
                     这是必需的对照——若随机 POI 也几乎总是 2 跳可达，则"存在路径"
                     本身不构成解释力，必须诚实报告。
  - dist_hit vs dist_miss : 命中样本与未命中样本的 d(H, target) 分布及 Mann-Whitney U 检验，
                     检验"KG 距离近"是否确实与"模型预测正确"相关。
  - spearman(rank, dist)  : 每个样本内，模型给出的全候选排名与 KG 跳数的 Spearman 相关，
                     衡量「解释路径」与「模型决策」是否一致（解释-预测一致性）。
                     该值不度量因果，仅度量一致性，论文中据实表述。

用法
----
  python -m llm_stkg.explain --load_model _best_seed42.pt --out explain_report.json \
      --use_bge --use_sgcp --scorer dot --session_pool mean --sem_thr 0.90 \
      --max_degree 10 --device cpu --cases 8
"""
import argparse
import json
import os
import random
from collections import Counter, deque

import numpy as np
import torch

from .config import Config
from .data.foursquare_loader import load_real_nyc
from .model.stkg_net import STKGNet
from .train import TrajDataset, _collate


# --------------------------------------------------------------------------
# 图结构：把四类边合并为一张 CSR 多关系图（保留每条边的类型标签）
# --------------------------------------------------------------------------
class MultiRelGraph:
    def __init__(self, edge_index, num_nodes):
        rel_names, src, dst, rel = [], [], [], []
        for ri, (t, ei) in enumerate(sorted(edge_index.items())):
            rel_names.append(t)
            e = np.asarray(ei)
            if e.size == 0:
                continue
            if e.shape[0] != 2 and e.shape[1] == 2:
                e = e.T
            src.append(e[0]); dst.append(e[1])
            rel.append(np.full(e.shape[1], ri, dtype=np.int8))
        self.rel_names = rel_names
        self.n = num_nodes
        if not src:
            self.indptr = np.zeros(num_nodes + 1, dtype=np.int64)
            self.indices = np.empty(0, dtype=np.int64)
            self.etype = np.empty(0, dtype=np.int8)
            return
        s = np.concatenate(src); d = np.concatenate(dst); r = np.concatenate(rel)
        # 对称化（kg_builder 已双向，但同质并集里稳妥起见再补一次并去重）
        s2 = np.concatenate([s, d]); d2 = np.concatenate([d, s]); r2 = np.concatenate([r, r])
        key = s2.astype(np.int64) * num_nodes * 8 + d2.astype(np.int64) * 8 + r2
        _, uniq = np.unique(key, return_index=True)
        s2, d2, r2 = s2[uniq], d2[uniq], r2[uniq]
        order = np.argsort(s2, kind="stable")
        s2, d2, r2 = s2[order], d2[order], r2[order]
        self.indices = d2.astype(np.int64)
        self.etype = r2.astype(np.int8)
        cnt = np.bincount(s2, minlength=num_nodes)
        self.indptr = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)

    def neighbors_of_set(self, nodes):
        """向量化取一个节点集合的全部邻居（含重复）。"""
        starts = self.indptr[nodes]
        ends = self.indptr[nodes + 1]
        cnt = ends - starts
        total = int(cnt.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        base = np.repeat(starts, cnt)
        off = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        return self.indices[base + off]

    def bfs_dist(self, sources, max_depth=3):
        """多源 BFS，返回 int8 距离数组（-1 = 超出 max_depth 或不可达）。"""
        dist = np.full(self.n, -1, dtype=np.int8)
        src = np.unique(np.asarray(sources, dtype=np.int64))
        src = src[(src >= 0) & (src < self.n)]
        if src.size == 0:
            return dist
        dist[src] = 0
        frontier = src
        for d in range(1, max_depth + 1):
            nbr = self.neighbors_of_set(frontier)
            if nbr.size == 0:
                break
            nbr = np.unique(nbr)
            new = nbr[dist[nbr] < 0]
            if new.size == 0:
                break
            dist[new] = d
            frontier = new
        return dist

    def shortest_path(self, sources, target, max_depth=4):
        """回溯一条真实最短路径，返回 [(node, rel_name_into_node), ...]，首元素 rel=None。"""
        src = set(int(x) for x in sources if 0 <= int(x) < self.n)
        if not src:
            return None
        if target in src:
            return [(target, None)]
        parent = {s: (None, None) for s in src}
        q = deque((s, 0) for s in src)
        while q:
            u, d = q.popleft()
            if d >= max_depth:
                continue
            for k in range(self.indptr[u], self.indptr[u + 1]):
                v = int(self.indices[k])
                if v in parent:
                    continue
                parent[v] = (u, self.rel_names[int(self.etype[k])])
                if v == target:
                    path, cur = [], v
                    while cur is not None:
                        p, r = parent[cur]
                        path.append((cur, r))
                        cur = p
                    return list(reversed(path))
                q.append((v, d + 1))
        return None


# --------------------------------------------------------------------------
def load_model_and_scores(args, cfg):
    from .head_to_head import build_kg, build_ui_edge, build_pop_prior
    from .train import _build_samples

    pois, checkins, test_samples, num_pois, stats, _ = load_real_nyc(args.processed_dir, 0.0)
    print(f"[data] POI={num_pois} 训练轨迹={len(checkins)} 测试样本={len(test_samples)}")
    users = list({u for u, _ in checkins})
    n_users = max(users) + 1
    kg = build_kg(cfg, pois, checkins)
    print("[KG] 边统计:", kg.stats())

    ui_edge = build_ui_edge(checkins, num_pois) if getattr(cfg, "use_ui_graph", True) else None
    # pop_prior 必须传：C6 启用时 pop_feat 是 registered buffer，结构须与 checkpoint 一致
    model = STKGNet(cfg, num_pois, kg.num_cats, kg.cat_ids, kg.sem_vecs, kg.edge_index,
                    n_users=n_users, user_item_edge=ui_edge,
                    pop_prior=build_pop_prior(checkins, num_pois)).to(args.device)
    if args.load_model and os.path.exists(args.load_model):
        sd = torch.load(args.load_model, map_location=args.device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[warn] state_dict 不完全匹配 missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
        print(f"[model] 已加载权重 {args.load_model}")
    else:
        # 无权重时：训练一遍（保证解释报告与稿件报告的是同一模型族）
        print(f"[model] 未提供权重，就地训练 {cfg.epochs} epoch ...")
        from .head_to_head import train_ours
        train_samples = _build_samples(checkins, cfg.seq_len, set(users))
        model = train_ours(cfg, pois, checkins, train_samples, num_pois, args.device,
                           n_users=n_users, user_item_edge=ui_edge)
        if args.save_model:
            torch.save(model.state_dict(), args.save_model)
            print(f"[model] 已保存 -> {args.save_model}")

    # 全候选打分
    model.eval()
    from torch.utils.data import DataLoader
    ds = TrajDataset(test_samples, num_pois, cfg.neg_samples, random.Random(cfg.seed))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate)
    all_scores, all_tgt = [], []
    with torch.no_grad():
        for H, T, C, Y, U, TGT in dl:
            H, T, U = H.to(args.device), T.to(args.device), U.to(args.device)
            cand = torch.arange(num_pois).unsqueeze(0).expand(H.size(0), -1).to(args.device)
            all_scores.append(model(H, T, cand, U).cpu())
            all_tgt.append(TGT)
    scores = torch.cat(all_scores).numpy()
    tgts = torch.cat(all_tgt).numpy()
    return pois, checkins, test_samples, num_pois, kg, scores, tgts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--processed_dir", default=None)
    ap.add_argument("--load_model", default=None)
    ap.add_argument("--save_model", default=None)
    ap.add_argument("--out", default="explain_report.json")
    ap.add_argument("--cases", type=int, default=8, help="导出的真实解释案例条数")
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--n_eval", type=int, default=0, help="0=全部测试样本；>0 抽样加速")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    # 结构参数需与被解释模型一致
    ap.add_argument("--use_bge", action="store_true")
    ap.add_argument("--use_sgcp", action="store_true")
    ap.add_argument("--scorer", default="dot", choices=["mlp", "dot", "attn"])
    ap.add_argument("--session_pool", default="mean", choices=["gru", "mean"])
    ap.add_argument("--sem_thr", type=float, default=0.90)
    ap.add_argument("--sem_feat_mode", default="bge", choices=["bge", "none", "cat_onehot"])
    ap.add_argument("--max_degree", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--bge_model_dir", default="bge_model")
    ap.add_argument("--bge_cache", default="poi_bge_emb.npy")
    ap.add_argument("--prior_channels", default="",
                    help="C6 先验通道，须与待加载 checkpoint 的训练配置一致（如 'cnt,rec,pop'）")
    ap.add_argument("--gate_mode", default="off", choices=["context", "global", "off"],
                    help="C6 门控模式，须与待加载 checkpoint 一致")
    ap.add_argument("--no_kg_channel", action="store_true",
                    help="与 checkpoint 一致的 KG 通道开关（仅在该 checkpoint 训练时移除过 KG 通道才需要）")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    cfg = Config()
    cfg.seed = args.seed; cfg.device = args.device; cfg.epochs = args.epochs
    cfg.scorer = args.scorer; cfg.session_pool = args.session_pool
    cfg.semantic_sim_thr = args.sem_thr; cfg.max_degree = args.max_degree
    cfg.batch_size = args.batch_size; cfg.lr = args.lr
    cfg.sem_feat_mode = args.sem_feat_mode
    if args.use_bge:
        cfg.use_bge = True; cfg.bge_model_dir = args.bge_model_dir
        cfg.bge_cache = args.bge_cache; cfg.sem_dim = 768
    if args.use_sgcp:
        cfg.use_sgcp = True
    cfg.prior_channels = args.prior_channels
    cfg.gate_mode = args.gate_mode
    if args.no_kg_channel:
        cfg.use_kg_channel = False

    pois, checkins, test_samples, num_pois, kg, scores, tgts = load_model_and_scores(args, cfg)
    G = MultiRelGraph(kg.edge_index, num_pois)
    print(f"[graph] 合并多关系图: |V|={G.n} |E(有向)|={G.indices.size} "
          f"平均度={G.indices.size / max(G.n, 1):.1f} 关系={G.rel_names}")

    # 训练频次（用于 Popularity 对照）
    freq = Counter(p for _, seq in checkins for p in seq)
    pop_top1 = max(range(num_pois), key=lambda p: freq.get(p, 0))

    idxs = list(range(len(test_samples)))
    if args.n_eval and args.n_eval < len(idxs):
        idxs = random.Random(args.seed).sample(idxs, args.n_eval)
    print(f"[explain] 分析样本数: {len(idxs)}")

    rng = random.Random(args.seed)
    D = args.max_depth
    stat = {k: Counter() for k in ["top1", "target", "random", "pop"]}
    hit_dist, miss_dist, spearmans = [], [], []
    first_rel = Counter()
    cases, fail_cases = [], []
    # 重访统计：session-based next-POI 数据里相当一部分 target 直接出现在 history 中。
    # 若不单独报告，"KG 路径解释"会被这部分平凡样本（d=0，无需任何路径）拉高，
    # 因此下面把 novel（target ∉ history）子集单独统计，作为解释力的严格口径。
    revisit_n, novel_n = 0, 0
    stat_novel = {k: Counter() for k in ["top1", "target"]}
    order_all = np.argsort(-scores, axis=1)

    for n, i in enumerate(idxs):
        _, hist, tgt = test_samples[i]
        hist = [int(h) for h in hist if 0 <= int(h) < num_pois]
        if not hist:
            continue
        dist = G.bfs_dist(hist, max_depth=D)
        top1 = int(order_all[i, 0])
        rnd = rng.randrange(num_pois)
        for key, node in [("top1", top1), ("target", int(tgt)), ("random", rnd), ("pop", pop_top1)]:
            d = int(dist[node])
            stat[key][d if d >= 0 else 99] += 1

        dt = int(dist[int(tgt)])
        (hit_dist if int(tgt) in order_all[i, :10] else miss_dist).append(dt if dt >= 0 else D + 1)

        # novel 子集：target 不在 history 中（真正需要"推荐新地点"的样本）
        is_novel = int(tgt) not in set(hist)
        if is_novel:
            novel_n += 1
            for key, node in [("top1", top1), ("target", int(tgt))]:
                d = int(dist[node])
                stat_novel[key][d if d >= 0 else 99] += 1
        else:
            revisit_n += 1

        # 解释-预测一致性：模型排名 vs KG 跳数 的 Spearman（仅在可达节点上算，避免 -1 主导）
        reach = np.where(dist >= 0)[0]
        if reach.size >= 30:
            rank_of = np.empty(num_pois, dtype=np.int32)
            rank_of[order_all[i]] = np.arange(num_pois)
            from scipy.stats import spearmanr
            rho = spearmanr(rank_of[reach], dist[reach]).statistic
            if not np.isnan(rho):
                spearmans.append(float(rho))

        # 首跳关系类型（top-1 是否为历史的直接邻居，经由哪类边）
        if int(dist[top1]) == 1:
            rels = set()
            for h in hist:
                for k in range(G.indptr[h], G.indptr[h + 1]):
                    if int(G.indices[k]) == top1:
                        rels.add(G.rel_names[int(G.etype[k])])
            for r in rels:
                first_rel[r] += 1

        # 导出真实案例。只取 novel 样本（target 不在 history），否则解释路径长度为 0 而平凡。
        if is_novel:
            rank_t = int(np.where(order_all[i] == int(tgt))[0][0])
            if len(cases) < args.cases and rank_t == 0:
                path = G.shortest_path(hist, top1, max_depth=D + 1)
                if path and len(path) >= 2:
                    cases.append({
                        "test_index": i,
                        "history_tail": [pois[h]["text"] for h in hist[-3:]],
                        "target_text": pois[int(tgt)]["text"],
                        "target_train_freq": int(freq.get(int(tgt), 0)),
                        "rank_of_target": rank_t + 1,
                        "path": [{"poi": pois[p]["text"], "via": r} for p, r in path],
                        "path_len_hops": len(path) - 1,
                    })
            # 失败案例：target 排名很差，但 top-1 仍能给出一条"看起来合理"的路径——
            # 这恰恰说明路径解释是事后合理化而非因果依据，必须在论文中一并展示。
            elif len(fail_cases) < max(2, args.cases // 3) and rank_t > 100 and top1 != int(tgt):
                path = G.shortest_path(hist, top1, max_depth=D + 1)
                if path and len(path) >= 2:
                    fail_cases.append({
                        "test_index": i,
                        "history_tail": [pois[h]["text"] for h in hist[-3:]],
                        "true_target_text": pois[int(tgt)]["text"],
                        "rank_of_target": rank_t + 1,
                        "recommended_text": pois[top1]["text"],
                        "path_to_recommended": [{"poi": pois[p]["text"], "via": r} for p, r in path],
                        "kg_hops_to_true_target": (int(dist[int(tgt)])
                                                   if int(dist[int(tgt)]) >= 0 else None),
                    })

    def pct(c):
        tot = sum(c.values()) or 1
        return {(f"{k}hop" if k != 99 else f">{D}hop_or_unreachable"): round(v / tot, 4)
                for k, v in sorted(c.items())}

    def cum_reach(c, L):
        tot = sum(c.values()) or 1
        return round(sum(v for k, v in c.items() if k <= L) / tot, 4)

    from scipy.stats import mannwhitneyu
    mw = None
    if len(hit_dist) > 10 and len(miss_dist) > 10:
        u, p = mannwhitneyu(hit_dist, miss_dist, alternative="less")
        mw = {"U": float(u), "p_value": float(p),
              "mean_dist_hit@10": round(float(np.mean(hit_dist)), 3),
              "mean_dist_miss@10": round(float(np.mean(miss_dist)), 3),
              "n_hit": len(hit_dist), "n_miss": len(miss_dist),
              "alternative": "hit 的 KG 跳数更小"}

    report = {
        "setting": {
            "n_samples_analyzed": len(idxs),
            "max_depth": D,
            "graph": {"nodes": int(G.n), "directed_edges": int(G.indices.size),
                      "relations": G.rel_names,
                      "avg_degree": round(G.indices.size / max(G.n, 1), 2)},
            "model": {"scorer": cfg.scorer, "session_pool": cfg.session_pool,
                      "sem_thr": cfg.semantic_sim_thr, "max_degree": cfg.max_degree,
                      "sem_feat_mode": cfg.sem_feat_mode,
                      "weights": args.load_model or "(trained in-place)"},
        },
        "revisit_vs_novel": {
            "n_revisit(target in history)": revisit_n,
            "n_novel(target not in history)": novel_n,
            "revisit_ratio": round(revisit_n / max(revisit_n + novel_n, 1), 4),
            "note": "重访样本的 KG 路径长度恒为 0，解释平凡；novel 子集才是解释力的严格口径",
        },
        "hop_distribution": {k: pct(v) for k, v in stat.items()},
        "reach_at_L": {k: {f"reach@{L}": cum_reach(v, L) for L in (1, 2, 3)}
                       for k, v in stat.items()},
        "reach_at_L_novel_only": {k: {f"reach@{L}": cum_reach(v, L) for L in (1, 2, 3)}
                                  for k, v in stat_novel.items()},
        "first_hop_relation_of_top1": dict(first_rel.most_common()),
        "explanation_prediction_consistency": {
            "spearman_rank_vs_kg_hops_mean": (round(float(np.mean(spearmans)), 4)
                                              if spearmans else None),
            "spearman_std": round(float(np.std(spearmans)), 4) if spearmans else None,
            "n": len(spearmans),
            "note": "正值 = 模型排名靠前的 POI 在 KG 上离历史更近（解释与决策一致）；"
                    "仅为一致性度量，非因果证据",
        },
        "kg_distance_vs_correctness": mw,
        "cases": cases,
        "failure_cases": fail_cases,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== KG 路径可达性（占比）===")
    hdr = f"{'目标类型':<10}{'reach@1':>10}{'reach@2':>10}{'reach@3':>10}"
    print(hdr); print("-" * len(hdr))
    for k in ["top1", "target", "pop", "random"]:
        r = report["reach_at_L"][k]
        print(f"{k:<10}{r['reach@1']:>10.4f}{r['reach@2']:>10.4f}{r['reach@3']:>10.4f}")
    print(f"\n重访/新地点: {report['revisit_vs_novel']}")
    print("=== 仅 novel 子集（target 不在 history）===")
    for k in ["top1", "target"]:
        r = report["reach_at_L_novel_only"][k]
        print(f"{k:<10}{r['reach@1']:>10.4f}{r['reach@2']:>10.4f}{r['reach@3']:>10.4f}")
    print(f"\n首跳关系分布(top-1 为 1 跳邻居时): {dict(first_rel.most_common())}")
    print(f"解释-预测一致性 Spearman: {report['explanation_prediction_consistency']}")
    print(f"KG 距离 vs 命中: {mw}")
    print(f"\n导出真实案例 {len(cases)} 条 -> {args.out}")
    for c in cases[:3]:
        chain = "  ".join(f"--{s['via']}-->  {s['poi']}" if s["via"] else s["poi"]
                          for s in c["path"])
        print(f"  [#{c['test_index']}] {chain}")


if __name__ == "__main__":
    main()
