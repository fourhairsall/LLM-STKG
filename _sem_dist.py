import numpy as np
from llm_stkg.data.foursquare_loader import load_real_nyc
from llm_stkg.kg.bge_encoder import BGESemanticEncoder

pois, checkins, test_samples, num_pois, stats, cold = load_real_nyc(None, 0.0)
texts = [p["text"] for p in pois]
print("n_pois=", len(texts), "sample=", texts[0], flush=True)
enc = BGESemanticEncoder("bge_model")
V = enc.encode(texts)
V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
S = V @ V.T
iu = np.triu_indices(len(texts), k=1)
vals = S[iu]
for q in [50, 80, 90, 95, 98, 99]:
    print("pct%d sim=%.4f" % (q, float(np.percentile(vals, q))), flush=True)
print("pairs=%d" % len(vals), flush=True)
for t in [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]:
    n = int((vals >= t).sum())
    print("thr=%.2f edges=%d (%.3f%%)" % (t, n, 100.0 * n / len(vals)), flush=True)
