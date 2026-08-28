"""为新增文本域（Steam-200k / Amazon Beauty）生成 BGE 语义嵌入缓存。

与 prepare_tky_bge.py 同构：读取 processed/pois.json 的 text 字段，用本地
bge_model (BAAI/bge-base-en-v1.5) 编码，输出 [num_pois, 768] L2 归一化矩阵。
仅供给 LLM4POI-style 基线做 POI token 语义种子；ours 跨域仍走 w/o LLM-text 模式。

注意：BGE(sentence-transformers/torch) 编码含 matmul，必须带线程前缀防沙箱 segfault。
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np
from llm_stkg.kg.llm_interface import LLMInterface

BGE_DIR = os.path.join(HERE, "bge_model")


def gen(processed_dir, out_cache):
    pois = json.load(open(os.path.join(processed_dir, "pois.json"), encoding="utf-8"))
    texts = [str(m.get("text", "")) for m in pois]
    print(f"[bge] {os.path.basename(out_cache)}: {len(texts)} POIs", flush=True)
    print(f"      sample text: {texts[0][:70]!r}", flush=True)
    llm = LLMInterface(bge_model_dir=BGE_DIR)
    vecs = llm.text_embedding(texts)          # List[List[float]]，顺序与 pois 连续索引一致
    arr = np.asarray(vecs, dtype=np.float64)
    print(f"[bge] raw shape={arr.shape}", flush=True)
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
    np.save(out_cache, arr)
    print(f"[bge] saved -> {out_cache}  shape={arr.shape}", flush=True)


if __name__ == "__main__":
    gen(os.path.join(ROOT, "data", "steam200k", "processed"),
        os.path.join(HERE, "poi_bge_emb_steam200k.npy"))
    gen(os.path.join(ROOT, "data", "amazon_beauty", "processed"),
        os.path.join(HERE, "poi_bge_emb_amazonbeauty.npy"))
