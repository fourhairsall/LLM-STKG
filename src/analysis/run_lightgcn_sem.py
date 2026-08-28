"""LightGCN + LLM 语义特征增强 —— 与基线 LightGCN 同协议公平对照。

动机（判废 v2 之后）：用户要求「在现有表现最好的模型基础上改进」，经确认选 LightGCN
作为最强骨干，注入 BGE/LLM 语义 item 特征（语义初始化 / 语义残差），而非再堆 LLM-KG 图。

对照协议（与 baseline_ranks.py 严格一致，否则不公平）：
  - 数据：load_real_nyc（真实 Foursquare-NYC，POI 连续重映射 0..N-1）
  - 样本：_build_samples(hist_mode="user", seq_len=200)，与官方测试协议一致
  - 训练：BPR（与基线 LightGCN 完全一致），epochs/lr/bs 同基线
  - 评估：全候选 session_predict 打分 + rank_metrics / rank_diag（同 honest_eval）
  - 冷启动子集：训练频次 <= 2 的 POI 作为目标样本的额外报告（语义先验应在此受益）

输出 lightgcn_sem.json（含基线 LightGCN 对照 + 各模式 full/cold 指标与 rank_diag）。
"""
import argparse
import json
import os
import sys
import time

# ---- 环境铁律：6 线程前缀，避免 torch 偶发 segfault ----
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_stkg.data.foursquare_loader import load_real_nyc   # noqa: E402
from llm_stkg.train import _build_samples                   # noqa: E402
from llm_stkg.evaluate import rank_metrics, rank_diag       # noqa: E402
from llm_stkg.lightgcn_sem import LightGCNSem               # noqa: E402
from llm_stkg.baselines import LightGCN as VanillaLightGCN  # noqa: E402


def build_scores(model, test_samples):
    """全候选打分（emb 仅算一次后复用，避免 1447 次重复 GNN 传播）。"""
    with torch.no_grad():
        emb = model._emb().to("cpu")                        # [I, dim]
        scs = []
        for _, h, _ in test_samples:
            if not h:
                h = [0]
            hv = emb[torch.tensor(h, dtype=torch.long)].mean(0)
            scs.append(hv @ emb.T)
    return torch.stack(scs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--modes", default="vanilla,resid,resid1,init,freeze_sem")
    ap.add_argument("--out", default="lightgcn_sem.json")
    ap.add_argument("--baseline_json", default="baseline_ranks.json")
    ap.add_argument("--processed_dir", default=None,
                    help="real_foursquare_nyc/processed 目录；缺省自动探测")
    ap.add_argument("--smoke", action="store_true",
                    help="用小子集 + cpu + 1 epoch 验证协议对齐（不写最终产物）")
    args = ap.parse_args()

    # 探测真实数据目录：loader 默认相对路径当前为空，回退到项目根 data/
    if args.processed_dir is None:
        _here = os.path.dirname(os.path.abspath(__file__))
        _cand = [
            os.path.normpath(os.path.join(
                _here, "..", "..", "data", "real_foursquare_nyc", "processed")),
            os.path.normpath(os.path.join(
                _here, "..", "..", "..", "data", "real_foursquare_nyc", "processed")),
        ]
        for _c in _cand:
            if os.path.exists(os.path.join(_c, "poi_meta.json")):
                args.processed_dir = _c
                break
    print(f"[data] processed_dir={args.processed_dir}", flush=True)

    torch.manual_seed(args.seed)
    pois, checkins, test_samples, num_pois, stats, _ = load_real_nyc(
        args.processed_dir, 0.0)
    users = list({u for u, _ in checkins})
    n_users = max(users) + 1
    train_samples = _build_samples(checkins, args.seq_len, set(users),
                                   hist_mode="user")
    sem = np.load("poi_bge_emb.npy")
    assert sem.shape[0] == num_pois, f"sem {sem.shape} vs num_pois {num_pois}"

    hl = sum(len(h) for _, h, _ in train_samples) / max(1, len(train_samples))
    rv = sum(1 for _, h, t in train_samples if t in set(h)) / max(1, len(train_samples))
    print(f"[data] num_pois={num_pois} n_users={n_users} "
          f"n_test={len(test_samples)} n_train={len(train_samples)}")
    print(f"[samples] hist_mode=user seq_len={args.seq_len} "
          f"hist_len_mean={hl:.1f} revisit_ratio={rv:.4f}")

    tgts = torch.tensor([int(t) for _, _, t in test_samples], dtype=torch.long)

    # ---- 冷启动子集：训练频次 <= 2 的 POI 作为目标 ----
    from collections import Counter
    freq = Counter(p for _, seq in checkins for p in seq)
    cold_set = {p for p in range(num_pois) if freq.get(p, 0) <= 2}

    if args.smoke:
        train_samples = train_samples[:2000]
        test_samples = test_samples[:200]
        tgts = tgts[:200]
        args.epochs = 1
        args.device = "cpu"
        args.modes = "vanilla,resid,resid1,init,freeze_sem"
        print("[smoke] reduced to", len(train_samples), "train /",
              len(test_samples), "test, cpu, 1 epoch")

    cold_idx = [i for i, (_, _, t) in enumerate(test_samples) if int(t) in cold_set]
    print(f"[cold] freq<=2 POIs={len(cold_set)} test_samples_in_cold={len(cold_idx)}")

    results, diags = {}, {}
    for spec in args.modes.split(","):
        spec = spec.strip()
        if spec == "vanilla":
            # 同协议真基线：直接用 baselines.LightGCN（nn.Embedding 默认 init std=1），
            # 用于隔离"V init 差异 + 残差几何"对 resid 结果的混淆。
            key = "LightGCN(vanilla, same-protocol)"
            print(f"\n=== {key} ===", flush=True)
            torch.manual_seed(args.seed)
            model = VanillaLightGCN(num_pois, dim=64, n_layers=3)
            t0 = time.time()
            model.fit(train_samples, epochs=args.epochs, lr=1e-3, bs=1024,
                      device=args.device)
            scores = build_scores(model, test_samples)
            res = rank_metrics(scores, tgts)
            diag = rank_diag(scores, tgts)
            res_cold = (rank_metrics(scores[cold_idx], tgts[cold_idx])
                        if cold_idx else None)
            results[key] = res
            diags[key] = {"full": diag, "cold": res_cold, "n_cold": len(cold_idx)}
            print(f"[{key}] full   = {res}")
            if res_cold:
                print(f"[{key}] cold(n={len(cold_idx)}) = {res_cold}")
            print(f"[{key}] time={time.time()-t0:.1f}s", flush=True)
            if args.smoke:
                continue
            _dump(args.out, num_pois, len(test_samples), train_samples, hl, rv,
                  results, diags, args, stats)
            continue
        # ---- 语义增强变体 ----
        if spec == "freeze_sem":
            mode, freeze, vinit, key = "resid", True, 0.1, "LightGCN+Sem(freeze=sem)"
        elif spec == "init":
            mode, freeze, vinit, key = "init", False, 0.1, "LightGCN+Sem(init)"
        elif spec == "resid1":   # 与 vanilla 唯一差异=残差项，V init 也用 std=1 对齐
            mode, freeze, vinit, key = "resid", False, 1.0, "LightGCN+Sem(resid,v1)"
        else:
            mode, freeze, vinit, key = "resid", False, 0.1, "LightGCN+Sem(resid)"
        print(f"\n=== {key} (mode={mode}, freeze_sem={freeze}, v_init={vinit}) ===",
              flush=True)
        torch.manual_seed(args.seed)
        model = LightGCNSem(num_pois, sem, dim=64, n_layers=3,
                            mode=mode, freeze_sem=freeze, v_init=vinit)
        t0 = time.time()
        model.fit(train_samples, epochs=args.epochs, lr=1e-3, bs=1024,
                  device=args.device)
        scores = build_scores(model, test_samples)
        res = rank_metrics(scores, tgts)
        diag = rank_diag(scores, tgts)
        res_cold = (rank_metrics(scores[cold_idx], tgts[cold_idx])
                    if cold_idx else None)
        results[key] = res
        diags[key] = {"full": diag, "cold": res_cold, "n_cold": len(cold_idx)}
        print(f"[{key}] full   = {res}")
        if res_cold:
            print(f"[{key}] cold(n={len(cold_idx)}) = {res_cold}")
        print(f"[{key}] time={time.time()-t0:.1f}s", flush=True)
        if args.smoke:
            continue
        # 增量落盘
        _dump(args.out, num_pois, len(test_samples), train_samples, hl, rv,
              results, diags, args, stats)

    # 载入基线 LightGCN 对照
    if not args.smoke and os.path.exists(args.baseline_json):
        base = json.load(open(args.baseline_json, encoding="utf-8"))
        bl = base.get("results", {})
        for k in ("LightGCN", "BPR-MF", "Popularity (Pop)"):
            if k in bl:
                results[k] = bl[k]
        diags["LightGCN"] = base.get("rank_diag", {}).get("full", {})
        _dump(args.out, num_pois, len(test_samples), train_samples, hl, rv,
              results, diags, args, stats)
        print("\n=== 对照表（full-candidate）===")
        print(f"{'model':28s} {'R@5':>7s} {'R@10':>7s} {'N@10':>7s}")
        order = ["Popularity (Pop)", "BPR-MF", "LightGCN",
                 "LightGCN(vanilla, same-protocol)",
                 "LightGCN+Sem(resid)", "LightGCN+Sem(resid,v1)",
                 "LightGCN+Sem(init)", "LightGCN+Sem(freeze=sem)"]
        for k in order:
            if k in results:
                r = results[k]
                print(f"{k:28s} {r['Recall@5']:7.4f} {r['Recall@10']:7.4f} "
                      f"{r['NDCG@10']:7.4f}")
    print(f"\n[saved] {args.out}")


def _dump(out, num_pois, n_test, train_samples, hl, rv, results, diags, args, stats):
    json.dump({
        "dataset": "Foursquare-NYC (LLM4POI, real)",
        "num_test": n_test, "num_pois": num_pois,
        "epochs": args.epochs, "seed": args.seed,
        "protocol": {"hist_mode": "user", "seq_len": args.seq_len,
                     "n_train_samples": len(train_samples),
                     "hist_len_mean": round(hl, 1), "revisit_ratio": round(rv, 4)},
        "stats": stats,
        "results": results, "rank_diag": diags,
    }, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
