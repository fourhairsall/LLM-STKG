# -*- coding: utf-8 -*-
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "ML-1M FULLBASE (switch OFF, prop ON)": "sw_ml1m_FULLBASE_ep20.json",
    "ML-1M no_graph  (switch ON,  prop OFF)": "sw_ml1m_no_graph_ep20.json",
    "Steam FULLBASE (switch OFF, prop ON)": "sw_steam_FULLBASE_ep20.json",
    "Steam nograph   (switch ON,  prop OFF)": "sw_steam_nograph.json",
}

def flatten(o, pre="", depth=0):
    """打印 dict 的所有叶子/短值，限制深度。"""
    if depth > 3:
        return
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)):
                print(f"{pre}{k}: {type(v).__name__}({len(v)})")
                flatten(v, pre + "  ", depth + 1)
            else:
                s = str(v)
                print(f"{pre}{k}: {s[:120]}")
    elif isinstance(o, list):
        print(f"{pre}[list len={len(o)}]")
        if o and isinstance(o[0], dict):
            print(f"{pre}  elem0 keys: {list(o[0].keys())[:20]}")
        elif o:
            print(f"{pre}  sample: {str(o[:3])[:120]}")

for label, fn in FILES.items():
    p = os.path.join(HERE, fn)
    print("=" * 78)
    print(label, "->", fn, "exists=", os.path.exists(p))
    if not os.path.exists(p):
        continue
    d = json.load(open(p, encoding="utf-8"))
    # 顶层键
    if isinstance(d, dict):
        print("TOP-LEVEL KEYS:", list(d.keys()))
        # 尝试直接打印已知指标键
        for mk in ("R@10", "R@5", "NDCG@10", "NDCG@5", "recall@10", "ndcg@10",
                   "recall", "ndcg", "metrics", "result", "final", "best",
                   "poi_cosine", "poi_grad", "cosine", "grad_norm", "collapse"):
            if mk in d:
                print(f"  [{mk}] = {str(d[mk])[:160]}")
    else:
        print("TYPE:", type(d).__name__, "len", len(d))
        if d and isinstance(d[0], dict):
            print("elem0 keys:", list(d[0].keys())[:20])
    # 浅层展开（仅一层）
    print("--- shallow walk ---")
    flatten(d, "  ", 0 if not isinstance(d, dict) else 1)
