"""为 Foursquare-TKY 生成 BGE 语义嵌入缓存 (poi_bge_emb_tky.npy)。

与 llm_stkg.head_to_head 的主流程严格对齐：
  - 用 load_real_nyc(processed_dir=.../real_foursquare_tky/processed) 取得 pois 列表，
    其内部已按 text = f"{cat_name} near {lat:.2f},{lng:.2f}" 构造 POI 文本（与 NYC 完全一致）；
  - 用本地 bge_model (BAAI/bge-base-en-v1.5) 编码，输出 [num_pois, 768] 的 L2 归一化矩阵；
  - 存为 poi_bge_emb_tky.npy，**不覆盖** NYC 的 poi_bge_emb.npy。

运行前需先跑 real_data_prepare_tky.py 生成 processed/。
BGE 编码走 sentence_transformers(PyTorch)，必须带线程前缀防沙箱 segfault。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # .../旅游推荐论文
WORKSPACE = os.path.dirname(ROOT)                  # .../2026年7月
sys.path.insert(0, HERE)

from llm_stkg.data.foursquare_loader import load_real_nyc
from llm_stkg.kg.llm_interface import LLMInterface

PROCESSED = os.path.join(WORKSPACE, "data", "real_foursquare_tky", "processed")
OUT_CACHE = os.path.join(HERE, "poi_bge_emb_tky.npy")
BGE_DIR = os.path.join(HERE, "bge_model")


def main():
    print(f"[tky-bge] 加载 TKY processed: {PROCESSED}", flush=True)
    pois, _, _, num_pois, stats, _ = load_real_nyc(PROCESSED)
    print(f"[tky-bge] POI 数 = {num_pois}", flush=True)
    texts = [m["text"] for m in pois]
    print(f"[tky-bge] 示例文本: {texts[:3]}", flush=True)

    print(f"[tky-bge] 加载 BGE 编码器: {BGE_DIR}", flush=True)
    llm = LLMInterface(bge_model_dir=BGE_DIR)
    vecs = llm.text_embedding(texts)  # List[List[float]], 顺序与 pois 连续索引一致
    import numpy as np
    arr = np.asarray(vecs, dtype=np.float64)
    print(f"[tky-bge] 嵌入矩阵: {arr.shape}", flush=True)
    # L2 归一化（与 build_kg 内部 sem_norm 一致，双保险）
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
    np.save(OUT_CACHE, arr)
    print(f"[tky-bge] 已保存 TKY BGE 缓存 -> {OUT_CACHE}  shape={arr.shape}", flush=True)


if __name__ == "__main__":
    main()
