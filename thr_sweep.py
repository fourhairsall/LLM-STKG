"""阈值敏感性扫描驱动器（P1-7 实证版）。

动机：P1-7 审稿意见要求"重训多个阈值以证伪 net-zero 对阈值稳健"，而非仅用单阈值
(0.90) 的 SGCP-off 逻辑论证。本驱动器在**生产 NYC 协议**下（bs1024/lr4e-3/ep30/
use_bge/use_sgcp/C6=cnt,rec,pop+context/max_degree=10，与 c6_runs.py 完全一致，
唯一变量是 --sem_thr）扫 6 个 BGE 语义相似度阈值，每阈值 1 个种子（seed 42，与
p1_5_ablation 定位协议一致），收集每个阈值下的 R@10 / N@10 / 冷启动 / 语义边数。

若 R@10 在 6 个阈值下均落在 0.644–0.648 噪声带内（与 0.90 基准 0.6475 无显著差异），
则直接实证"语义边数量/阈值对排名 net-zero"，封堵"只在一个阈值验过"的质疑。

用法（后台）：
  python thr_sweep.py --workers 6
增量写 thr_sweep.json；已完成任务跳过，进程被杀可续跑。
"""
import os
import json
import time
import argparse
import subprocess
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"C:\Users\Lenovo\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
MASTER = os.path.join(HERE, "thr_sweep.json")

THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "TORCH_NUM_THREADS": "1",
}

SEED = 42
THRESHOLDS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]


def build_jobs(bs, lr, epochs, hist_mode="user", seq_len=200):
    """唯一变量是 --sem_thr。其余与生产 ours (c6_full) 完全一致。"""
    common = ["--device", "cuda", "--max_degree", "10",
              "--batch_size", str(bs), "--lr", str(lr), "--epochs", str(epochs),
              "--bge_model_dir", "bge_model", "--bge_cache", "poi_bge_emb.npy",
              "--use_bge",
              "--processed_dir", "D:/databuddy/专利写作/2026年7月/data/real_foursquare_nyc/processed",
              "--scorer", "dot", "--session_pool", "mean", "--use_sgcp", "--ours_only",
              "--hist_mode", hist_mode, "--seq_len", str(seq_len),
              "--prior_channels", "cnt,rec,pop", "--gate_mode", "context",
              "--seed", str(SEED)]
    jobs = []
    for thr in THRESHOLDS:
        jobs.append((f"thr_{thr:0.2f}_s{SEED}",
                     common + ["--sem_thr", f"{thr:.2f}"]))
    return jobs


def ts():
    return datetime.datetime.now().strftime("%m-%d %H:%M:%S")


def env():
    e = os.environ.copy()
    e.update(THREAD_ENV)
    return e


def load_master():
    if os.path.exists(MASTER):
        try:
            with open(MASTER, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"jobs": {}}


def save(m):
    tmp = MASTER + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    os.replace(tmp, MASTER)


def collect(name, out_json, rc, secs):
    rec = {"status": "ok" if rc == 0 else f"rc={rc}", "seconds": round(secs, 1),
           "out_json": os.path.basename(out_json), "finished": ts()}
    if rc == 0 and os.path.exists(out_json):
        try:
            d = json.load(open(out_json, encoding="utf-8"))
            rec["ours"] = d.get("results", {}).get("LLM-STKG (ours)")
            td = d.get("train_diag", {}) or {}
            rec["cfg"] = {k: td.get(k) for k in
                          ("semantic_sim_thr", "max_degree", "use_kg_channel",
                           "prior_channels", "gate_mode")}
            rec["kg_edges"] = td.get("kg_edges")
            rec["train_seconds"] = td.get("train_seconds")
            cs = d.get("cold_start(\u22645)", {})
            rec["cold_n"] = cs.get("n")
            rec["ours_cold"] = cs.get("results", {}).get("LLM-STKG (ours)")
        except Exception as e:
            rec["status"] = f"json_err:{e}"
    return rec


def main():
    global MASTER
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", default="4e-3")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default=None, help="逗号分隔，仅跑指定任务名")
    ap.add_argument("--hist_mode", default="user", choices=["trajectory", "user"])
    ap.add_argument("--seq_len", type=int, default=200)
    a = ap.parse_args()

    from _singleton import acquire
    acquire("_thr_sweep.lock", "thr_sweep.py")
    MASTER = os.path.join(HERE, "thr_sweep.json")

    jobs = build_jobs(a.bs, a.lr, a.epochs, a.hist_mode, a.seq_len)
    if a.only:
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if j[0] in keep]
    jobs = [(f"t_{n}", e) for n, e in jobs]

    m = load_master()
    m["config"] = {"bs": a.bs, "lr": a.lr, "epochs": a.epochs, "max_degree": 10,
                   "sem_thr_values": THRESHOLDS, "seed": SEED,
                   "note": "P1-7 empirical threshold sweep; only --sem_thr varies vs production ours",
                   "started": ts()}
    m.setdefault("jobs", {})
    pending = [j for j in jobs if m["jobs"].get(j[0], {}).get("status") != "ok"]
    print(f"[{ts()}] total={len(jobs)} pending={len(pending)} workers={a.workers}", flush=True)
    save(m)

    running = []
    queue = list(pending)
    while queue or running:
        while queue and len(running) < a.workers:
            name, extra = queue.pop(0)
            out_json = os.path.join(HERE, f"{name}.json")
            log = os.path.join(HERE, f"{name}.log")
            cmd = [PY, "-u", "-m", "llm_stkg.head_to_head"] + extra + ["--out", out_json]
            lf = open(log, "w", encoding="utf-8")
            p = subprocess.Popen(cmd, cwd=HERE, env=env(), stdout=lf, stderr=subprocess.STDOUT)
            running.append((name, p, lf, out_json, time.time()))
            m["jobs"][name] = {"status": "running", "pid": p.pid, "started": ts()}
            save(m)
            print(f"[{ts()}] START {name} pid={p.pid} ({len(running)} running, {len(queue)} queued)",
                  flush=True)
        time.sleep(15)
        still = []
        for name, p, lf, out_json, t0 in running:
            if p.poll() is None:
                still.append((name, p, lf, out_json, t0))
                continue
            lf.close()
            rec = collect(name, out_json, p.returncode, time.time() - t0)
            m["jobs"][name] = rec
            save(m)
            o = rec.get("ours") or {}
            print(f"[{ts()}] DONE  {name} {rec['status']} {rec['seconds']}s "
                  f"R@10={o.get('Recall@10')} N@10={o.get('NDCG@10')} "
                  f"sem_edges={(rec.get('kg_edges') or {}).get('semantic')}", flush=True)
        running = still
    m["finished"] = ts()
    save(m)
    print(f"[{ts()}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
