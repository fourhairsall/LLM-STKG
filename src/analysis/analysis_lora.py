import numpy as np
f = "llm4poi_lora_open_llama_7b_v2_nyc_scores.npz"
d = np.load(f)
scores = d["scores"]; targets = d["targets"]
print("scores", scores.shape, "targets", targets.shape)
print(f"POI-logits mean={scores.mean():.4f} std={scores.std():.4f}  per-row std mean={scores.std(1).mean():.6f}")
ranks = np.array([int((scores[i] > scores[i][int(targets[i])]).sum()) + 1 for i in range(len(targets))])
print(f"mean_rank(strict>)={ranks.mean():.1f}  R@10(strict>)={(ranks<=10).mean():.4f}")
for i in [0,1,2,100,500]:
    top = np.argsort(-scores[i])[:5]
    print(f"  smp{i} tgt={int(targets[i])} top5_poi={top.tolist()} top5_logit={[round(float(scores[i][j]),3) for j in top]}")
