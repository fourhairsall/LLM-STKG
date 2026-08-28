"""C1 语义相似度质量分析 — 证明 LLM/bge 语义嵌入能区分语义相关 POI。
产出：semantic_analysis.json（统计量）+ semantic_sim_hist.png（双栏图）+ 终端打印 Top-3 近邻案例。
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EMB = "poi_bge_emb.npy"
META = "D:/databuddy/专利写作/2026年7月/data/real_foursquare_nyc/processed/poi_meta.json"
OUT_JSON = "semantic_analysis.json"
OUT_PNG = "semantic_sim_hist.png"

# 1) 加载嵌入并 L2 归一
emb = np.load(EMB).astype(np.float32)
print("emb shape:", emb.shape)
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
N = emb.shape[0]

# 2) 全对余弦相似度（N 小，直接全量）
S = emb @ emb.T                       # [N, N]
S = S - np.eye(N) * 2.0               # 去自相似

# 3) 相似度分布统计
allpairs = S.flatten()
allpairs = allpairs[allpairs > -2]    # 去掉对角占位
stats = {
    "n_pois": int(N),
    "cos_sim_mean": float(allpairs.mean()),
    "cos_sim_std": float(allpairs.std()),
    "cos_sim_p05": float(np.percentile(allpairs, 5)),
    "cos_sim_p25": float(np.percentile(allpairs, 25)),
    "cos_sim_p50": float(np.percentile(allpairs, 50)),
    "cos_sim_p75": float(np.percentile(allpairs, 75)),
    "cos_sim_p95": float(np.percentile(allpairs, 95)),
    # C1 语义边阈值 τ=0.30 下的"将连边"密度
    "edge_density_tau0.30": float((allpairs > 0.30).mean()),
    "edge_density_tau0.50": float((allpairs > 0.50).mean()),
    "edge_density_tau0.85": float((allpairs > 0.85).mean()),
    "edge_density_tau0.90": float((allpairs > 0.90).mean()),
    "edge_density_tau0.92": float((allpairs > 0.92).mean()),
    "frac_dup_ge0.99": float((allpairs >= 0.99).mean()),
}

# 4) 同近邻类目重合率（验证语义质量 + emb 与 poi_meta 对齐）
meta = json.load(open(META))
cat_of = {int(k): v.get("cat_name", "?") for k, v in meta.items()}

# 4b) 同类 vs 跨类 相似度对比（抽样，作为 C1 语义边质量的真证据）
rng2 = np.random.RandomState(1)
sa, cr = [], []
samp = rng2.choice(N, size=min(4000, N), replace=False)
for a in samp:
    ca = cat_of.get(int(a), "?")
    for b in rng2.choice(N, size=30, replace=False):
        if a == b:
            continue
        cb = cat_of.get(int(b), "?")
        (sa if ca == cb else cr).append(float(S[a, b]))
stats["same_cat_mean_sim"] = float(np.mean(sa)) if sa else None
stats["cross_cat_mean_sim"] = float(np.mean(cr)) if cr else None
stats["same_minus_cross_cat"] = (stats["same_cat_mean_sim"] - stats["cross_cat_mean_sim"]) if sa and cr else None
# 对齐 sanity: 抽样 2000 个 POI 的 top1 近邻同 cat 比例
rng = np.random.RandomState(0)
idx = rng.choice(N, size=min(2000, N), replace=False)
topk = 1
order = np.argsort(-S[idx], axis=1)[:, :topk]
same_cat = 0
for i, js in zip(idx, order):
    ci = cat_of.get(int(i), "?")
    for j in js:
        if cat_of.get(int(j), "?") == ci:
            same_cat += 1
stats["top1_same_cat_frac(sanity)"] = same_cat / len(idx) / topk

# 5) Top-3 近邻案例（选若干代表性 POI：不同类目、不同相似度档）
examples = []
cand_cats = ["Hotel", "Bar", "Restaurant", "Museum", "Park", "Subway"]
for cat in cand_cats:
    pid = next((p for p, c in cat_of.items() if c == cat), None)
    if pid is None or pid >= N:
        continue
    row = S[pid]
    js = np.argsort(-row)[:4]  # 含自身，取后 3 个非自身
    neigh = []
    for j in js:
        if int(j) == pid:
            continue
        neigh.append({"poi": int(j), "cat": cat_of.get(int(j), "?"),
                      "cos": round(float(row[j]), 4)})
        if len(neigh) == 3:
            break
    examples.append({
        "query_poi": int(pid), "query_cat": cat,
        "neighbors": neigh
    })

# 6) 出图
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].hist(allpairs, bins=60, color="#3b7dd8", edgecolor="white")
axes[0].axvline(0.85, color="#d9534f", ls="--", lw=1.5, label="C1 edge tau=0.85")
axes[0].axvline(stats["cos_sim_mean"], color="#5cb85c", ls="-", lw=1.5,
                label=f"mean={stats['cos_sim_mean']:.3f}")
axes[0].set_title("bge cosine similarity over all POI pairs (N=%d)" % N)
axes[0].set_xlabel("cosine similarity"); axes[0].set_ylabel("POI pair count")
axes[0].legend(fontsize=8)

ks = [1, 2, 3, 5, 10]
same_frac = []
for k in ks:
    o = np.argsort(-S[idx], axis=1)[:, :k]
    sf = 0
    for i, js in zip(idx, o):
        ci = cat_of.get(int(i), "?")
        sf += sum(1 for j in js if cat_of.get(int(j), "?") == ci)
    same_frac.append(sf / len(idx) / k)
axes[1].plot(ks, same_frac, "o-", color="#e08e0b")
axes[1].set_title("Top-k nearest neighbor same-category ratio")
axes[1].set_xlabel("k (neighbor rank)"); axes[1].set_ylabel("same-category ratio")
axes[1].set_ylim(0, 1)
for x, y in zip(ks, same_frac):
    axes[1].annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 6), fontsize=8)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=130)
print("saved", OUT_PNG)

out = {"stats": stats, "top3_examples": examples}
json.dump(out, open(OUT_JSON, "w"), ensure_ascii=False, indent=2)
print(json.dumps(stats, ensure_ascii=False, indent=2))
print("=== Top-3 语义近邻案例 ===")
for ex in examples:
    print(f"  [{ex['query_cat']}] POI#{ex['query_poi']} -> " +
          ", ".join(f"#{n['poi']}({n['cat']},{n['cos']})" for n in ex["neighbors"]))
