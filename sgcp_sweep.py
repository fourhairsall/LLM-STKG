"""SGCP 调参 + 多种子显著性验证驱动器。

两阶段（单一后台进程，增量写 sgcp_sweep.json，会话重置仍存活）：
  阶段① 调参(tuning)：固定 seed=42，扫描 sgcp_scale ∈ {3,5,8}（ours_only 加速），
            选 ours NDCG@10 最高的 scale（已知 scale=1.0 基线 N@10=0.2558 可直接纳入比较）。
  阶段② 显著性(significance)：用最佳 scale，跑 5 个种子（全量含 LightGCN 同协议对比），
            得 ours vs LightGCN 的 mean±std，检验 NDCG 是否越过 LightGCN。

用法（后台）：
  python sgcp_sweep.py
"""
import os
import sys
import json
import time
import subprocess
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"C:\Users\Lenovo\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
MASTER = os.path.join(HERE, "sgcp_sweep.json")

# 线程隔离（防 OpenBLAS/MKL 抢线程 segfault）
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "TORCH_NUM_THREADS": "1",
}

# 已知 scale=1.0 / seed=42 的真实结果（来自 head_to_head_c1_sgcp.json），直接纳入调参比较
PREEXISTING_SCALE1 = {
    "seed": 42, "status": "preexisting",
    "ours": {"Recall@5": 0.4264, "NDCG@5": 0.2373, "Recall@10": 0.5612, "NDCG@10": 0.2558},
}

TUNING_SCALES = [3.0, 5.0, 8.0]      # scale=1.0 已知，不再重跑
BIAS = 3.0
SEEDS_SIG = [42, 123, 777, 2024, 99]


def _env():
    e = os.environ.copy()
    e.update(THREAD_ENV)
    return e


def _ours_only_cmd(out_json, scale, bias, seed):
    return [
        PY, "-m", "llm_stkg.head_to_head",
        "--use_bge", "--use_sgcp", "--scorer", "dot", "--session_pool", "mean",
        "--sem_thr", "0.90", "--bge_model_dir", "bge_model", "--bge_cache", "poi_bge_emb.npy",
        "--device", "cpu", "--epochs", "30",
        "--seed", str(seed), "--sgcp_scale", str(scale), "--sgcp_bias", str(bias),
        "--ours_only", "--out", out_json,
    ]


def _full_cmd(out_json, scale, bias, seed):
    return [
        PY, "-m", "llm_stkg.head_to_head",
        "--use_bge", "--use_sgcp", "--scorer", "dot", "--session_pool", "mean",
        "--sem_thr", "0.90", "--bge_model_dir", "bge_model", "--bge_cache", "poi_bge_emb.npy",
        "--device", "cpu", "--epochs", "30",
        "--seed", str(seed), "--sgcp_scale", str(scale), "--sgcp_bias", str(bias),
        "--out", out_json,
    ]


def _run(label, cmd, log_path):
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, cwd=HERE, env=_env(), stdout=lf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    out_json = cmd[cmd.index("--out") + 1]
    rec = {"label": label, "status": "ok" if rc.returncode == 0 else f"rc={rc.returncode}",
           "seconds": round(dt, 1), "out_json": os.path.basename(out_json)}
    if rc.returncode == 0 and os.path.exists(out_json):
        try:
            with open(out_json, encoding="utf-8") as f:
                payload = json.load(f)
            r = payload.get("results", {})
            if "LLM-STKG (ours)" in r:
                rec["ours"] = r["LLM-STKG (ours)"]
            if "LightGCN" in r:
                rec["LightGCN"] = r["LightGCN"]
            cs = payload.get("cold_start(≤5)", {}).get("results", {})
            if "LLM-STKG (ours)" in cs:
                rec["ours_cold"] = cs["LLM-STKG (ours)"]
        except Exception as e:
            rec["status"] = f"json_err:{e}"
    else:
        rec["status"] = f"rc={rc.returncode}"
    return rec


def _load_master():
    if os.path.exists(MASTER):
        try:
            with open(MASTER, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"tuning": {}, "significance": {}}


def _save_master(m):
    with open(MASTER, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)


def main():
    m = _load_master()
    m.setdefault("tuning", {})
    m.setdefault("significance", {})

    # ---------- 阶段① 调参 ----------
    print(f"[{_ts()}] === 阶段① 调参：scale∈{TUNING_SCALES}, seed=42, ours_only ===", flush=True)
    m["tuning"]["scale=1.0"] = PREEXISTING_SCALE1   # 已知基线
    for scale in TUNING_SCALES:
        label = f"tune_s{scale}"
        out_json = os.path.join(HERE, f"_sweep_{label}.json")
        log_path = os.path.join(HERE, f"_sweep_{label}.log")
        print(f"[{_ts()}] 启动 {label} (scale={scale}) ...", flush=True)
        rec = _run(label, _ours_only_cmd(out_json, scale, BIAS, 42), log_path)
        m["tuning"][f"scale={scale}"] = rec
        _save_master(m)
        print(f"[{_ts()}] {label} 完成: status={rec['status']} ours_N@10="
              f"{rec.get('ours', {}).get('NDCG@10')}", flush=True)

    # 选最佳 scale（ours NDCG@10 最大）
    best_scale = max(
        [1.0] + TUNING_SCALES,
        key=lambda s: m["tuning"].get(f"scale={s}", {}).get("ours", {}).get("NDCG@10", -1),
    )
    m["best_scale"] = best_scale
    _save_master(m)
    print(f"[{_ts()}] === 阶段① 结束：最佳 scale={best_scale} ===", flush=True)

    # ---------- 阶段② 显著性 ----------
    print(f"[{_ts()}] === 阶段② 显著性：best_scale={best_scale}, seeds={SEEDS_SIG}, 全量 ===", flush=True)
    for seed in SEEDS_SIG:
        label = f"sig_s{int(best_scale)}_seed{seed}"
        out_json = os.path.join(HERE, f"_sweep_{label}.json")
        log_path = os.path.join(HERE, f"_sweep_{label}.log")
        print(f"[{_ts()}] 启动 {label} ...", flush=True)
        rec = _run(label, _full_cmd(out_json, best_scale, BIAS, seed), log_path)
        m["significance"][f"seed={seed}"] = rec
        _save_master(m)
        print(f"[{_ts()}] {label} 完成: status={rec['status']} ours_N@10="
              f"{rec.get('ours', {}).get('NDCG@10')} vs LightGCN_N@10="
              f"{rec.get('LightGCN', {}).get('NDCG@10')}", flush=True)

    # 汇总显著性
    ours_n5 = [m["significance"][k]["ours"]["NDCG@5"] for k in m["significance"] if "ours" in m["significance"][k]]
    ours_n10 = [m["significance"][k]["ours"]["NDCG@10"] for k in m["significance"] if "ours" in m["significance"][k]]
    lg_n5 = [m["significance"][k]["LightGCN"]["NDCG@5"] for k in m["significance"] if "LightGCN" in m["significance"][k]]
    lg_n10 = [m["significance"][k]["LightGCN"]["NDCG@10"] for k in m["significance"] if "LightGCN" in m["significance"][k]]

    def _mean(x): return round(sum(x) / len(x), 4) if x else None
    def _std(x):
        if not x: return None
        mu = sum(x) / len(x)
        return round((sum((v - mu) ** 2 for v in x) / len(x)) ** 0.5, 4)

    m["summary"] = {
        "n_seeds": len(ours_n10),
        "ours_NDCG@5_mean": _mean(ours_n5), "ours_NDCG@5_std": _std(ours_n5),
        "ours_NDCG@10_mean": _mean(ours_n10), "ours_NDCG@10_std": _std(ours_n10),
        "lgcn_NDCG@5_mean": _mean(lg_n5), "lgcn_NDCG@5_std": _std(lg_n5),
        "lgcn_NDCG@10_mean": _mean(lg_n10), "lgcn_NDCG@10_std": _std(lg_n10),
        "ours_beats_lgcn_N@10": _mean(ours_n10) is not None and _mean(lg_n10) is not None and _mean(ours_n10) > _mean(lg_n10),
    }
    _save_master(m)
    print(f"[{_ts()}] === 全部完成。ours_N@10 mean={m['summary']['ours_NDCG@10_mean']}±"
          f"{m['summary']['ours_NDCG@10_std']} vs LightGCN {m['summary']['lgcn_NDCG@10_mean']}±"
          f"{m['summary']['lgcn_NDCG@10_std']} ===", flush=True)


def _ts():
    return datetime.datetime.now().strftime("%m-%d %H:%M:%S")


if __name__ == "__main__":
    main()
