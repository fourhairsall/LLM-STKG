"""对比诊断输出：ours 与基线（尤其 LightGCN）的正样本排名分布。

用法:
  python analyze_diag.py _diag.json
读取 payload["rank_diag"]["full"] = {模型名: {pct_rank1, median_rank, rank_hist, ...}}
重点看 ours 的 pct_rank1 / median_rank 是否显著低于 LightGCN —— 若是，则 NDCG 钝的根因是
"正样本常落在 rank 5~10 而非 rank 1"，与 R@K 高（正样本在 top-K 内）但 NDCG 低吻合。
"""
import json, sys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "_diag.json"
    d = json.load(open(path, encoding="utf-8"))
    sec = d.get("rank_diag", {}).get("full", {})
    if not sec:
        print("未找到 rank_diag.full 段"); return
    keys = ["pct_rank1", "median_rank", "top10_score_std", "target_vs_top1_ratio"]
    hist_keys = ["rank1", "rank2-3", "rank4-5", "rank6-10", "rank11-50", "rank51-200", "rank200+"]
    # 表头
    header = f"{'Model':<20}" + "".join(f"{k:>14}" for k in keys)
    print(header)
    print("-" * len(header))
    rows = {}
    for name, m in sec.items():
        rows[name] = m
        print(f"{name:<20}" + "".join(f"{m.get(k,0):>14.4f}" for k in keys))
    print("\n=== 排名分布直方图 (越高=正样本越靠后) ===")
    hhead = f"{'Model':<20}" + "".join(f"{k:>12}" for k in hist_keys)
    print(hhead)
    print("-" * len(hhead))
    for name, m in sec.items():
        h = m.get("rank_hist", {})
        print(f"{name:<20}" + "".join(f"{h.get(k,0):>12.4f}" for k in hist_keys))
    # 结论提示
    ours = rows.get("LLM-STKG (ours)")
    lgc = rows.get("LightGCN")
    if ours and lgc:
        print("\n=== 诊断结论提示 ===")
        print(f"ours  pct_rank1={ours.get('pct_rank1'):.4f}  median_rank={ours.get('median_rank')}")
        print(f"LGCN  pct_rank1={lgc.get('pct_rank1'):.4f}  median_rank={lgc.get('median_rank')}")
        if ours.get('pct_rank1', 0) < lgc.get('pct_rank1', 1):
            print("→ ours 把正样本排到第 1 名的比例更低 → NDCG 钝主因=正样本未进榜首（排名靠后），非召回失败。")
        else:
            print("→ ours 与 LGCN 的 rank1 率接近 → NDCG 差异或来自 top-10 分数离散度/校准。")

if __name__ == "__main__":
    main()
