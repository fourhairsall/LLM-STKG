"""受控验证：HF/历史频率基线强势是否依赖高重访率。

生成不同 revisit_ratio 的 NYC 风格合成数据（revisit_ratio=每步重访最近 POI 的概率），
对每组跑零参数规则基线（HF/HR/MC1/Pop），观察 HF@10 是否随重访率单调上升。
若高重访率下 HF 强、低重访率下 HF 弱，则证明「HF 接近天花板」是 revisit 主导基准
（如真实 Foursquare-NYC 75.7% revisit）的特性，而非 POI 推荐普适规律。
替代不可行的 TKY 跨城市验证。
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from llm_stkg.data.foursquare_loader import generate_foursquare_like


def build_test(checkins, train_ratio=0.8):
    tests = []
    for uid, seq in checkins:
        if len(seq) < 6:
            continue
        k = max(2, int(len(seq) * train_ratio))
        for i in range(k, len(seq)):
            tests.append((uid, seq[:i], seq[i]))
    return tests


def eval_trivial(tests, num_pois, checkins):
    n = len(tests)
    pop = np.zeros(num_pois)
    for uid, seq in checkins:
        pop[np.array(seq, dtype=int)] += 1
    pop = pop / (pop.sum() + 1e-9)
    HF = np.zeros((n, num_pois), dtype=np.float32)
    HR = np.zeros((n, num_pois), dtype=np.float32)
    MC1 = np.zeros((n, num_pois), dtype=np.float32)
    tgts = np.zeros(n, dtype=np.int64)
    for i, (uid, hist, tgt) in enumerate(tests):
        c = Counter(hist)
        for p, cnt in c.items():
            HF[i, p] = cnt
        if hist:
            HR[i, hist[-1]] = 1.0
        if len(hist) >= 2:
            MC1[i, hist[-2]] = 1.0
        tgts[i] = tgt
    res = {}
    for name, S in [('HF', HF), ('HR', HR), ('MC1', MC1),
                    ('Pop', np.tile(pop, (n, 1)).astype(np.float32))]:
        ts = S[np.arange(n), tgts]
        gt = (S > ts[:, None]).sum(axis=1) + 1
        res[name] = gt
    return res


print(f"{'rev_ratio':>9} {'obs_rev':>8} {'HF@10':>8} {'HR@10':>8} {'MC1@10':>8} {'Pop@10':>8}")
for rv in [0.1, 0.3, 0.5, 0.7, 0.9]:
    t0 = time.time()
    pois, checkins = generate_foursquare_like(
        num_users=800, num_pois=1500, seq_min=20, seq_max=50,
        seed=42, revisit_ratio=rv)
    num_pois = len(pois)
    tests = build_test(checkins)
    obs = np.mean([1.0 if t in h else 0.0 for _, h, t in tests])
    ranks = eval_trivial(tests, num_pois, checkins)
    r10 = {k: float(np.mean(ranks[k] <= 10)) for k in ranks}
    print(f"{rv:>9.1f} {obs:>8.3f} {r10['HF']:>8.4f} {r10['HR']:>8.4f} "
          f"{r10['MC1']:>8.4f} {r10['Pop']:>8.4f}  ({time.time()-t0:.1f}s)")
