"""跨数据集对比表汇总器。

把多个 `head_to_head.py --dataset X --out X.json` 的结果聚合成一张论文/专利可直接引用的
Markdown + LaTeX 表，并强制随表输出三项"防误读"元信息：

  1. **text_mode**：该数据集是否使用了 LLM 文本语义。Foursquare-NYC 有 POI 名称/类目文本，
     可跑完整 LLM-STKG；MovieLens/Gowalla/Steam 无（或仅有弱）物品文本，统一以
     **w/o LLM-text** 模式评测（语义特征与语义边关闭，仅保留行为先验 C6 + 共现 cooc +
     可选的地理/类目边）。不标注这一点，跨域数字会被误读成"完整方法的效果"。
  2. **revisit_ratio_test**：测试目标是否已出现在该样本历史中的比例。Foursquare-NYC≈0.757，
     "复读历史"即可拿到高 Recall；MovieLens/Steam≈0（无重复消费）。这是判断
     History-Freq / History-Recency 这类零学习基线是否构成有效对照的前提统计量。
  3. **num_pois / num_test**：候选空间与测试规模。全候选排名下 Recall 与候选数强相关，
     跨数据集横向比较绝对值无意义，只能比"同一数据集内的相对排序"。

用法
----
  python cross_dataset_table.py \
      --run "Foursquare-NYC=baseline_ranks.json" \
      --run "MovieLens-1M=ml1m_head2head.json" \
      --run "Gowalla=gowalla_head2head.json" \
      --run "Steam=steam_head2head.json" \
      --md cross_dataset_table.md --tex cross_dataset_table.tex \
      --json cross_dataset_report.json
"""
import argparse
import json
import os

# 表内模型显示顺序（缺失的自动跳过；未列出的追加在末尾）
MODEL_ORDER = [
    "LLM-STKG (ours)",
    "LM-STKG (ours)",
    "SASRec",
    "eSASRec",
    "eSASRec-CE",
    "LightGCN",
    "BPR-MF",
    "FPMC",
    "GRU-STGN",
    "History-Freq (HF)",
    "History-Recency (HR)",
    "Markov-1",
    "Popularity (Pop)",
]

METRICS = ["Recall@5", "Recall@10", "NDCG@5", "NDCG@10"]


def load_run(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def order_models(names):
    """按 MODEL_ORDER 排序；未登记的模型保持原相对顺序追加在后面。"""
    known = [m for m in MODEL_ORDER if m in names]
    rest = [m for m in names if m not in MODEL_ORDER]
    return known + rest


def fmt(v):
    return "—" if v is None else f"{v:.4f}"


def build(runs):
    """runs: list[(label, payload)] -> (meta_rows, model_rows, all_models)"""
    meta_rows = []
    for label, p in runs:
        meta_rows.append({
            "dataset": label,
            "full_name": p.get("dataset", label),
            "text_mode": p.get("text_mode", "n/a"),
            "num_pois": p.get("num_pois"),
            "num_test": p.get("num_test"),
            "revisit_test": p.get("revisit_ratio_test"),
            "hist_len_test": p.get("hist_len_mean_test"),
            "protocol": p.get("protocol", ""),
            # 历史屏蔽是跨域可比性的前提：无重复消费域（revisit≈0）若不屏蔽历史，
            # 已交互物品会挤占 top-K，所有模型被系统性压低（Steam 上实测使 ours 从
            # 高于零参数水位跌到其 0.39 倍）。必须随表披露，否则数字不可复核。
            "mask_history": p.get("mask_history"),
            "mask_history_mode": p.get("mask_history_mode"),
            # 训练样本预算：hist_mode=user 在稠密消费域生成数十万样本（ML-1M 466341），
            # 单卡跑不完 6 个模型，故按固定 seed 均匀下采样。ours 与全部基线共用同一批，
            # 列内公平不受影响，但必须披露否则读者无法复现绝对值。
            "train_sample_ratio": p.get("train_sample_ratio"),
            "train_samples_used": p.get("train_samples_used"),
            # KG 图传播是否启用：Steam 上共访图密度=1.000、枢纽垄断（邻居 Jaccard 0.55）
            # 导致表征塌缩（POI 两两余弦 0.99998），必须禁用传播，此为域适配而非调参。
            "no_graph": ((p.get("train_diag") or {}).get("no_graph")
                         if isinstance(p.get("train_diag"), dict) else None),
        })
    all_models = []
    for _, p in runs:
        for m in (p.get("results") or {}):
            if m not in all_models:
                all_models.append(m)
    all_models = order_models(all_models)

    model_rows = {}
    for m in all_models:
        model_rows[m] = {}
        for label, p in runs:
            r = (p.get("results") or {}).get(m)
            model_rows[m][label] = ({k: r.get(k) for k in METRICS} if r else None)
    return meta_rows, model_rows, all_models


def best_per_column(model_rows, labels, metric):
    """返回每个数据集在该指标上的最优值，用于加粗。"""
    best = {}
    for lab in labels:
        vals = [(m, model_rows[m][lab][metric])
                for m in model_rows
                if model_rows[m].get(lab) and model_rows[m][lab].get(metric) is not None]
        if vals:
            best[lab] = max(v for _, v in vals)
    return best


def to_md(meta_rows, model_rows, all_models, labels, metric_pair=("Recall@5", "Recall@10")):
    L = []
    L.append("# 跨数据集对比（全候选排名）\n")
    L.append("## 数据集与协议元信息\n")
    L.append("| 数据集 | 文本模式 | 候选 POI/物品数 | 测试样本数 | 测试端重访率 | 测试端历史均长 "
             "| 历史屏蔽 | 训练样本(采样比) | KG 传播 |")
    L.append("|---|---|---:|---:|---:|---:|:---:|---:|:---:|")
    for r in meta_rows:
        rv = "—" if r["revisit_test"] is None else f"{r['revisit_test']:.4f}"
        mh = r.get("mask_history")
        mh_s = "—" if mh is None else ("ON" if mh else "OFF")
        if r.get("mask_history_mode"):
            mh_s += f" ({r['mask_history_mode']})"
        _tsr = r.get("train_sample_ratio")
        _tsu = r.get("train_samples_used")
        ts_s = "—" if _tsu is None else (f"{_tsu}" + (f" ({_tsr:.2f})" if _tsr is not None
                                                      and _tsr < 0.999 else " (全量)"))
        _ng = r.get("no_graph")
        ng_s = "—" if _ng is None else ("关闭" if _ng else "启用")
        L.append(f"| {r['full_name']} | {r['text_mode']} | {r['num_pois']} | "
                 f"{r['num_test']} | {rv} | {r['hist_len_test']} | {mh_s} | {ts_s} | {ng_s} |")
    L.append("")
    L.append("> **历史屏蔽（mask_history）**：在无重复消费域（重访率≈0，目标必不在历史中），"
             "按 SASRec 标准协议屏蔽用户已交互物品后再排名；在重访主导域（Foursquare / Gowalla）"
             "则**不得**屏蔽，否则会把正确答案一并删除。本表由实测重访率自动判定（阈值 0.05），"
             "对 ours 与全部基线**统一施加**，保证同一列内公平。\n")
    L.append("> **重访率的含义**：测试目标已出现在该样本历史中的比例。Foursquare-NYC 高达 0.75+，"
             "「复读历史」类零学习基线即可取得高 Recall；MovieLens-1M / Steam 无重复消费（≈0），"
             "此类基线完全失效。因此跨数据集验证能够排除「收益来自重访红利」的替代解释。\n")
    L.append("> **不可横向比较绝对值**：全候选排名下 Recall 与候选空间大小强相关，"
             "跨数据集只应比较同一列内部的模型相对排序。\n")
    L.append("> **KG 传播列**：Steam 上共访图密度达 1.000、前 10 个枢纽物品垄断 77.6% 的边"
             "（随机节点对的 top-k 邻居 Jaccard 相似度 0.55），均值聚合后所有物品表征塌成一根"
             "（两两余弦 0.99998、嵌入梯度 8.45e-05、‖μ‖/𝔼‖δ‖=162.98）。此时关闭图传播使梯度回升约 580 倍"
             "（4.93e-02）、Recall@10 由 0.0809 升至 0.0985（+22%）。该开关由共访图统计量在训练前自动"
             "判定，属于**域适配规则**而非按测试集调参；判据与四数据集实测见 §5.10.3。\n")

    for metric in metric_pair:
        L.append(f"\n## {metric}\n")
        header = "| 模型 | " + " | ".join(labels) + " |"
        L.append(header)
        L.append("|---" * (len(labels) + 1) + "|")
        best = best_per_column(model_rows, labels, metric)
        for m in all_models:
            cells = []
            for lab in labels:
                d = model_rows[m].get(lab)
                if not d or d.get(metric) is None:
                    cells.append("—")
                else:
                    v = d[metric]
                    cells.append(f"**{v:.4f}**" if best.get(lab) == v else f"{v:.4f}")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
    L.append("")
    return "\n".join(L)


def to_tex(model_rows, all_models, labels, metric="Recall@10"):
    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{跨数据集全候选排名对比（%s）。MovieLens-1M / Gowalla / Steam 上物品无可用文本，"
             r"统一以 \emph{w/o LLM-text} 模式评测；全候选 Recall 与候选空间大小强相关，"
             r"仅同列内部可比。}" % metric.replace("@", "@"))
    L.append(r"\label{tab:cross-dataset}")
    L.append(r"\begin{tabular}{l" + "c" * len(labels) + "}")
    L.append(r"\toprule")
    L.append("模型 & " + " & ".join(labels) + r" \\")
    L.append(r"\midrule")
    best = best_per_column(model_rows, labels, metric)
    for m in all_models:
        cells = []
        for lab in labels:
            d = model_rows[m].get(lab)
            if not d or d.get(metric) is None:
                cells.append("--")
            else:
                v = d[metric]
                cells.append(r"\textbf{%.4f}" % v if best.get(lab) == v else "%.4f" % v)
        L.append(m.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help='形如 "显示名=结果json路径"，可重复。')
    ap.add_argument("--md", default="cross_dataset_table.md")
    ap.add_argument("--tex", default="cross_dataset_table.tex")
    ap.add_argument("--json", default="cross_dataset_report.json")
    args = ap.parse_args()

    runs, labels = [], []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run 需为 '名称=路径' 形式: {spec}")
        label, path = spec.split("=", 1)
        label, path = label.strip(), path.strip()
        if not os.path.exists(path):
            print(f"[skip] 结果文件不存在: {path}")
            continue
        runs.append((label, load_run(path)))
        labels.append(label)
    if not runs:
        raise SystemExit("没有可用的结果文件。")

    meta_rows, model_rows, all_models = build(runs)
    md = to_md(meta_rows, model_rows, all_models, labels)
    tex = to_tex(model_rows, all_models, labels)

    with open(args.md, "w", encoding="utf-8") as f:
        f.write(md)
    with open(args.tex, "w", encoding="utf-8") as f:
        f.write(tex)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta_rows,
                   "labels": labels,
                   "models": all_models,
                   "metrics": {m: model_rows[m] for m in all_models}},
                  f, indent=2, ensure_ascii=False)
    print(md)
    print(f"\n[out] {args.md} / {args.tex} / {args.json}")


if __name__ == "__main__":
    main()
