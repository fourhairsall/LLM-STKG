import numpy as np
f = "llm4poi_ptuning_open_llama_7b_v2_nyc_scores.npz"
d = np.load(f)
scores = d["scores"]; targets = d["targets"]
print("scores", scores.shape, "targets", targets.shape, "dtype", scores.dtype)
ranks = []
for i in range(len(targets)):
    sc = scores[i]; t = int(targets[i])
    ranks.append(int((sc > sc[t]).sum()) + 1)
ranks = np.array(ranks)
print(f"mean_rank={ranks.mean():.1f} median={np.median(ranks):.0f} max={ranks.max()}")
print(f"R@1={(ranks<=1).mean():.4f}  R@5={(ranks<=5).mean():.4f}  R@10={(ranks<=10).mean():.4f}")
print(f"POI-logits mean={scores.mean():.4f} std={scores.std():.4f}  per-row std mean={scores.std(1).mean():.4f}")
print("--- few samples: target + top5 POI + their logits ---")
for i in [0,1,2,100,500]:
    top = np.argsort(-scores[i])[:5]
    print(f"  smp{i} tgt={int(targets[i])} top5_poi={top.tolist()} top5_logit={[round(float(scores[i][j]),3) for j in top]}")
PYEOF=1
