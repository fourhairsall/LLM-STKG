"""并排实验：近期方法真实基线 vs LLM-STKG（同协议全候选排名）。

两类近期方法（论文 Related Work / Discussion 引用的最新工作）的可运行代理：
  (A) BGE-Sem 语义检索（LLM4POI / CoMaPOI 式"语义即检索"代理）：
      轨迹 BGE 嵌入 -> POI BGE 余弦排名。直接检验论文核心论点
      "LLM 语义向量作为 ranking 特征零增益 / 结构化先验才有效"。
        - mean-hist : 历史 POI 的 BGE 均值作为轨迹表示
        - last-POI  : 最后一个 POI 的 BGE 作为轨迹表示
  (B) Rotan（Feng et al., KDD'24，"Rotate Item-oriented Network"）旋转感知时序模型：
      每 POI 一个轴角旋转 R_i，预测 = R_{i_L} e_{i_L} 与候选点积；BPR 训练。

所有指标用 llm_stkg.evaluate 的 target_rank（悲观并列）/ metrics_from_ranks，
与既有 ours / LightGCN / SASRec 结果严格同协议。

用法：
  python side_by_side.py --city nyc
  python side_by_side.py --city tky
  python side_by_side.py --city both
"""
import os
import sys
import json
import argparse

# ---- 线程前缀（防 torch/OpenBLAS segfault）----
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORKSPACE = os.path.dirname(ROOT)
sys.path.insert(0, HERE)

from llm_stkg.data.foursquare_loader import load_real_nyc          # noqa: E402
from llm_stkg.evaluate import target_rank, metrics_from_ranks      # noqa: E402

K_LIST = (5, 10)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# (A) BGE 语义检索基线
# ============================================================
def bge_semantic_scores(test_samples, num_pois, bge, variant):
    """返回 [B, N] 余弦打分矩阵。bge: [N,768] L2 归一化 torch tensor。"""
    B = len(test_samples)
    S = torch.zeros(B, num_pois, dtype=torch.float32)
    for i, (_, hist, _) in enumerate(test_samples):
        h = [int(p) for p in hist if 0 <= int(p) < num_pois]
        if not h:
            continue  # 全 0 -> 悲观并列给所有 POI 同排名
        if variant == "last":
            v = bge[h[-1]].clone()
        else:  # mean-hist
            v = bge[h].mean(dim=0)
        v = v / (v.norm() + 1e-8)
        S[i] = bge @ v          # cosine（bge 已 L2 归一化）
    return S


# ============================================================
# (B) Rotan（轴角旋转 + BPR）
# ============================================================
class Rotan(torch.nn.Module):
    def __init__(self, num_pois, d=32):
        super().__init__()
        self.d = d
        self.E = torch.nn.Embedding(num_pois, d)
        # 每 POI 一个可学习（原始）矩阵 M_i，反对称化 S_i = M_i - M_i^T 后
        # 经 Cayley 变换得到旋转矩阵 R_i = (I+S_i)(I-S_i)^{-1} ∈ SO(d)。
        self.M = torch.nn.Embedding(num_pois, d * d)
        torch.nn.init.xavier_uniform_(self.E.weight)
        torch.nn.init.zeros_(self.M.weight)

    def _rotation(self, idx):
        """返回旋转矩阵 R [b,d,d]（Cayley 变换，SO(d)）。"""
        b = idx.size(0)
        M = self.M(idx).view(b, self.d, self.d)           # [b,d,d]
        S = M - M.transpose(1, 2)                          # 反对称
        I = torch.eye(self.d, device=M.device).unsqueeze(0).expand(b, self.d, self.d)
        # R = (I-S)^{-1}(I+S) == (I+S)(I-S)^{-1}（S 反对称时两式相等）
        R = torch.linalg.solve(I - S, I + S)
        return R

    def query(self, last_idx):
        """预测查询向量 q = R_{i_L} e_{i_L}。"""
        e = self.E(last_idx)                              # [.,d]
        R = self._rotation(last_idx)                      # [.,d,d]
        return (R @ e.unsqueeze(-1)).squeeze(-1)          # [.,d]


def build_bpr_pairs(checkins, num_pois, max_len=20):
    """从训练 checkins 生成 (last_item -> next) 对。"""
    pairs = []
    for _, seq in checkins:
        seq = [int(p) for p in seq if 0 <= int(p) < num_pois]
        if len(seq) < 2:
            continue
        for t in range(1, len(seq)):
            pairs.append((seq[t - 1], seq[t]))
    return pairs


def train_rotan(checkins, num_pois, epochs=15, lr=1e-3, batch=512, seed=42,
                neg=3, d=32):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Rotan(num_pois, d=d).to(DEVICE)
    pairs = build_bpr_pairs(checkins, num_pois)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bpr = torch.nn.LogSigmoid()
    n = len(pairs)
    for ep in range(epochs):
        rng = np.random.default_rng(seed + ep)
        idx = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, batch):
            b = idx[s:s + batch]
            last = torch.tensor([pairs[j][0] for j in b], dtype=torch.long, device=DEVICE)
            pos = torch.tensor([pairs[j][1] for j in b], dtype=torch.long, device=DEVICE)
            q = model.query(last)                          # [b,d]
            pos_sc = (q * model.E(pos)).sum(-1)            # [b]
            # 负采样
            neg_idx = torch.randint(0, num_pois, (len(b), neg), device=DEVICE)
            neg_sc = (q.unsqueeze(1) * model.E(neg_idx)).sum(-1)  # [b,neg]
            loss = -bpr(pos_sc.unsqueeze(1) - neg_sc).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[Rotan] ep{ep} loss={tot/len(b):.4f}" if len(b) else "", flush=True)
    return model.cpu()


def rotan_scores(test_samples, num_pois, model):
    model.eval()
    B = len(test_samples)
    S = torch.zeros(B, num_pois, dtype=torch.float32)
    with torch.no_grad():
        for i, (_, hist, _) in enumerate(test_samples):
            h = [int(p) for p in hist if 0 <= int(p) < num_pois]
            if not h:
                continue
            last = torch.tensor([h[-1]], dtype=torch.long)
            q = model.query(last)[0]                       # [d]
            S[i] = model.E.weight @ q
    return S


# ============================================================
# 主流程
# ============================================================
def run_city(city, bge_cache, processed_dir, out_json):
    print(f"\n========== {city.upper()} ==========", flush=True)
    pois, checkins, test_samples, num_pois, stats, _ = load_real_nyc(processed_dir)
    tgts = torch.tensor([int(t) for _, _, t in test_samples], dtype=torch.long)
    n_test = len(test_samples)
    print(f"[data] num_pois={num_pois} n_test={n_test} "
          f"n_train_traj={len(checkins)}", flush=True)

    bge = torch.tensor(np.load(bge_cache), dtype=torch.float32)  # [N,768]
    if bge.shape[0] != num_pois:
        raise SystemExit(f"BGE 缓存行数 {bge.shape[0]} != 加载 POI 数 {num_pois}，"
                         f"索引未对齐！")

    full = {}
    metrics = {}

    # ---- (A) BGE 语义检索 ----
    for variant, label in (("mean", "BGE-Sem (mean-hist)"),
                           ("last", "BGE-Sem (last-POI)")):
        S = bge_semantic_scores(test_samples, num_pois, bge, variant)
        rank = target_rank(S, tgts)
        m = metrics_from_ranks(rank.float())
        full[label] = {"ranks": [int(r) for r in rank.tolist()]}
        metrics[label] = m
        print(f"[{label}] R@5={m['Recall@5']} R@10={m['Recall@10']} "
              f"N@10={m['NDCG@10']} median_rank={float(rank.median()):.1f}", flush=True)

    # ---- (B) Rotan ----
    model = train_rotan(checkins, num_pois)
    S = rotan_scores(test_samples, num_pois, model)
    rank = target_rank(S, tgts)
    m = metrics_from_ranks(rank.float())
    full["Rotan (KDD'24)"] = {"ranks": [int(r) for r in rank.tolist()]}
    metrics["Rotan (KDD'24)"] = m
    print(f"[Rotan (KDD'24)] R@5={m['Recall@5']} R@10={m['Recall@10']} "
          f"N@10={m['NDCG@10']} median_rank={float(rank.median()):.1f}", flush=True)

    payload = {
        "city": city,
        "num_pois": num_pois,
        "n_test": n_test,
        "rank_diag": {"full": full},
        "metrics": metrics,
        "note": "全候选排名，未屏蔽历史（与 ours / LightGCN 同协议）；"
                "NYC 重访主导域不屏蔽历史。",
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[saved] {out_json}", flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["nyc", "tky", "both"], default="both")
    args = ap.parse_args()

    cfg = {
        "nyc": {
            "bge_cache": os.path.join(HERE, "poi_bge_emb.npy"),
            "processed_dir": os.path.join(WORKSPACE, "data", "real_foursquare_nyc", "processed"),
            "out_json": os.path.join(HERE, "side_by_side_nyc.json"),
        },
        "tky": {
            "bge_cache": os.path.join(HERE, "poi_bge_emb_tky.npy"),
            "processed_dir": os.path.join(WORKSPACE, "data", "real_foursquare_tky", "processed"),
            "out_json": os.path.join(HERE, "side_by_side_tky.json"),
        },
    }
    if args.city in ("nyc", "both"):
        run_city("nyc", **cfg["nyc"])
    if args.city in ("tky", "both"):
        run_city("tky", **cfg["tky"])


if __name__ == "__main__":
    main()
