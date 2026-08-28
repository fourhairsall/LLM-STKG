"""跨域新文本域（Steam-200k / Amazon Beauty）子表 LaTeX 抽取器。

读取该域的 head_to_head 结果（ours + 6 基线）与 LLM4POI-style 结果
（文本种子 + ID-only 因果对照），生成与 multi_dataset_paper.tex 中 tab:cross
同版式（纵向堆叠子表、4 指标、列内最优加粗、ours 用 cbest 着色）的两个子表块，
直接粘贴进 tab:cross 的 \\bottomrule 之前即可。

用法（在 code/ 目录，后台结果 JSON 就绪后）：
  python extract_cross_new.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# 两个新文本域：head_to_head(ours+6基线) + LLM4POI(text) + LLM4POI(ID-only)
DOMAINS = {
    "Steam-200k": {
        "hh": "head_to_head_steam200k.json",
        "text": "llm4poi_steam200k.json",
        "nobge": "llm4poi_steam200k_nobge.json",
    },
    "Amazon Beauty": {
        "hh": "head_to_head_amazonbeauty.json",
        "text": "llm4poi_amazonbeauty.json",
        "nobge": "llm4poi_amazonbeauty_nobge.json",
    },
}

# 固定模型顺序（与 tab:cross 既有子表风格一致；LLM4POI 两变体紧随 ours）
MODEL_ORDER = [
    "LLM-STKG (ours)",
    "LLM4POI-style (text)",
    "LLM4POI-style (ID-only, no text)",
    "SASRec", "LightGCN", "BPR-MF", "FPMC", "GRU-STGN", "Popularity (Pop)",
]
METRICS = ["Recall@5", "Recall@10", "NDCG@5", "NDCG@10"]

DISPLAY = {
    "LLM-STKG (ours)": r"\textcolor{cbest}{\textbf{LLM-STKG}}",
    "LLM4POI-style (text)": "LLM4POI-style (text)",
    "LLM4POI-style (ID-only, no text)": "LLM4POI-style (ID-only)",
    "SASRec": "SASRec",
    "LightGCN": "LightGCN",
    "BPR-MF": "BPR-MF",
    "FPMC": "FPMC",
    "GRU-STGN": "GRU-STGN",
    "Popularity (Pop)": "Popularity",
}


def load(p):
    with open(os.path.join(HERE, p), encoding="utf-8") as f:
        return json.load(f)


def section(disp, paths):
    miss = [k for k, v in paths.items() if not os.path.exists(os.path.join(HERE, v))]
    if miss:
        print("%% [SKIP] %s 缺少结果文件: %s（后台仍在跑？）" % (disp, miss))
        return
    hh = load(paths["hh"])
    llm_text = load(paths["text"])
    llm_nobge = load(paths["nobge"])
    results = hh.get("results", {})
    rev = hh.get("revisit_ratio_test")
    mh = "ON" if hh.get("mask_history") else "OFF"
    npois = hh.get("num_pois")
    ntest = hh.get("num_test")

    rowmetrics = {}
    for m in MODEL_ORDER:
        if m == "LLM-STKG (ours)":
            rowmetrics[m] = results.get("LLM-STKG (ours)")
        elif m == "LLM4POI-style (text)":
            rowmetrics[m] = llm_text.get("metrics")
        elif m == "LLM4POI-style (ID-only, no text)":
            rowmetrics[m] = llm_nobge.get("metrics")
        else:
            rowmetrics[m] = results.get(m)

    best = {k: None for k in METRICS}
    for m in MODEL_ORDER:
        d = rowmetrics.get(m)
        if not d:
            continue
        for k in METRICS:
            v = d.get(k)
            if v is None:
                continue
            if best[k] is None or v > best[k]:
                best[k] = v

    print(r"\midrule")
    print(r"\multicolumn{5}{c}{\textbf{%s} (rev. %.4f, mask %s)} \\"
          % (disp, rev if rev is not None else 0.0, mh))
    print(r"Model & R@5 & R@10 & N@5 & N@10 \\")
    print(r"\midrule")
    for m in MODEL_ORDER:
        d = rowmetrics.get(m)
        if not d:
            cells = ["--"] * 4
        else:
            cells = []
            for k in METRICS:
                v = d.get(k)
                if v is None:
                    cells.append("--")
                else:
                    s = "%.4f" % v
                    if best.get(k) is not None and abs(v - best[k]) < 1e-9:
                        s = r"\textbf{" + s + "}"
                    cells.append(s)
        print("%s & %s \\\\" % (DISPLAY[m], " & ".join(cells)))
    print("%% %s: num_pois=%s num_test=%s revisit=%.4f mask=%s"
          % (disp, npois, ntest, rev if rev is not None else 0.0, mh))


def main():
    for disp, paths in DOMAINS.items():
        section(disp, paths)
    print("%% --- end of auto-extracted new-domain sub-tables ---")


if __name__ == "__main__":
    main()
