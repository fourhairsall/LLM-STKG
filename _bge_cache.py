"""一次性把 4980 个 POI 文本编码为 BGE 768 维向量并缓存到 poi_bge_emb.npy。

设计要点（规避沙箱 segfault）：
- 显式 torch.set_num_threads(1)（比仅设环境变量更稳）；
- 分块编码（chunk=16），逐块 flush 进度，单块失败不影响整体可续跑；
- 结果落盘 poi_bge_emb.npy，训练管线直接 load，避免每次训练都跑 transformer 推理。
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

from llm_stkg.data.foursquare_loader import load_real_nyc
from llm_stkg.kg.bge_encoder import BGESemanticEncoder

CHUNK = 16
OUT = "poi_bge_emb.npy"


def main():
    pois, checkins, test_samples, num_pois, stats, cold = load_real_nyc(None, 0.0)
    texts = [p["text"] for p in pois]
    print("n_pois=", len(texts), flush=True)

    enc = BGESemanticEncoder("bge_model")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("bge_model")
    torch.set_num_threads(1)

    all_vecs = []
    for i in range(0, len(texts), CHUNK):
        chunk = texts[i:i + CHUNK]
        v = model.encode(chunk, normalize_embeddings=True, convert_to_numpy=True,
                         batch_size=CHUNK, show_progress_bar=False)
        all_vecs.append(np.asarray(v, dtype=np.float64))
        print("encoded %d/%d" % (min(i + CHUNK, len(texts)), len(texts)), flush=True)

    V = np.vstack(all_vecs)
    np.save(OUT, V)
    print("saved", OUT, V.shape, flush=True)

    # 顺便给出相似度分布，便于选语义边阈值
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    S = Vn @ Vn.T
    iu = np.triu_indices(len(texts), k=1)
    vals = S[iu]
    for q in [50, 80, 90, 95, 98, 99]:
        print("pct%d sim=%.4f" % (q, float(np.percentile(vals, q))), flush=True)
    for t in [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]:
        n = int((vals >= t).sum())
        print("thr=%.2f edges=%d (%.3f%%)" % (t, n, 100.0 * n / len(vals)), flush=True)


if __name__ == "__main__":
    main()
