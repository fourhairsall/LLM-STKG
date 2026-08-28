"""P0 修复后超参 pilot：并行跑 3 个短周期配置，确认 [2,E] 边索引修复 + k-NN 剪枝后
模型仍能训起来，并选出兼顾质量与耗时的 batch_size / lr / epochs。

增量写 p0_pilot.json（会话中断也能查）。
"""
import os
import json
import time
import subprocess
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"C:\Users\Lenovo\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
MASTER = os.path.join(HERE, "p0_pilot.json")

THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "TORCH_NUM_THREADS": "1",
}

BASE = ["--use_bge", "--use_sgcp", "--scorer", "dot", "--session_pool", "mean",
        "--sem_thr", "0.90", "--bge_model_dir", "bge_model", "--bge_cache", "poi_bge_emb.npy",
        "--max_degree", "10", "--device", "cpu", "--seed", "42", "--ours_only"]

JOBS = [
    ("bs256_lr2e3_ep12", ["--batch_size", "256", "--lr", "2e-3", "--epochs", "12"]),
    ("bs256_lr3e3_ep12", ["--batch_size", "256", "--lr", "3e-3", "--epochs", "12"]),
    ("bs1024_lr4e3_ep30", ["--batch_size", "1024", "--lr", "4e-3", "--epochs", "30"]),
]


def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def env():
    e = os.environ.copy()
    e.update(THREAD_ENV)
    return e


def save(m):
    with open(MASTER, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)


def main():
    from _singleton import acquire
    acquire("_p0_pilot.lock", "p0_pilot.py")
    m = {"started": ts(), "jobs": {}}
    procs = []
    for name, extra in JOBS:
        out_json = os.path.join(HERE, f"_pilot_{name}.json")
        log = os.path.join(HERE, f"_pilot_{name}.log")
        cmd = [PY, "-m", "llm_stkg.head_to_head"] + BASE + extra + ["--out", out_json]
        lf = open(log, "w", encoding="utf-8")
        p = subprocess.Popen(cmd, cwd=HERE, env=env(), stdout=lf, stderr=subprocess.STDOUT)
        procs.append((name, p, lf, out_json, time.time()))
        m["jobs"][name] = {"status": "running", "pid": p.pid}
        print(f"[{ts()}] launched {name} pid={p.pid}", flush=True)
    save(m)

    while procs:
        time.sleep(20)
        still = []
        for name, p, lf, out_json, t0 in procs:
            if p.poll() is None:
                still.append((name, p, lf, out_json, t0))
                continue
            lf.close()
            rec = {"status": "ok" if p.returncode == 0 else f"rc={p.returncode}",
                   "seconds": round(time.time() - t0, 1)}
            if p.returncode == 0 and os.path.exists(out_json):
                d = json.load(open(out_json, encoding="utf-8"))
                rec["ours"] = d["results"].get("LLM-STKG (ours)")
                rec["train_diag"] = d.get("train_diag")
                cs = d.get("cold_start(\u22645)", {}).get("results", {})
                rec["ours_cold"] = cs.get("LLM-STKG (ours)")
            m["jobs"][name] = rec
            save(m)
            print(f"[{ts()}] done {name}: {rec.get('status')} {rec.get('ours')}", flush=True)
        procs = still
    m["finished"] = ts()
    save(m)
    print(f"[{ts()}] ALL PILOTS DONE", flush=True)


if __name__ == "__main__":
    main()
