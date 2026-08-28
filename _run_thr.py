import numpy as np, sys, json, traceback
sys.path.insert(0, '.')
def main():
    from llm_stkg.data.foursquare_loader import load_real_nyc
    from llm_stkg.kg.bge_encoder import BGESemanticEncoder
    pois, checkins, test_samples, num_pois, stats, cold = load_real_nyc("D:/databuddy/专利写作/2026年7月/data/real_foursquare_nyc/processed", 0.0)
    texts = [p['text'] for p in pois]
    enc = BGESemanticEncoder('bge_model')
    V = enc.encode(texts)
    V = V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-8)
    S = V@V.T
    iu = np.triu_indices(len(texts),k=1)
    vals = S[iu]
    out = {}
    out['n_pois'] = len(texts)
    out['pct'] = {str(q): round(float(np.percentile(vals,q)),4) for q in [50,80,90,95,98,99]}
    out['pairs'] = int(len(vals))
    out['thr'] = {str(t): int((vals>=t).sum()) for t in [0.80,0.85,0.88,0.90,0.92,0.95]}
    with open('_threshold_scan.json','w') as f:
        json.dump(out, f, indent=1)
    print("DONE", json.dumps(out))
main()
