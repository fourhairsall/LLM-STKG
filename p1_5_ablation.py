"""P1-5 单边缘类型 + 传播深度消融驱动器（诚实修订）。

审稿人 P1-5 要求：补单边类型消融（geo-only/cat-only/sem-only/co-visit-only）+
传播深度消融（L=1 vs L=2），定位 full(0.6475) 与 no-KG(0.6448) 之间「0.0027 差异」
的来源，并解释 SGCP 在 NYC 为何惰性（inert）。

本驱动器复用 c6_runs.py 的生产 NYC 协议（bs1024/lr4e-3/ep30/use_bge sem_thr 0.90/
dot/mean/use_sgcp/cnt,rec,pop/context/hist_mode user/seq_len 200）——唯一的变量是单
边缘类型开关或 num_gnn_layers。续跑框架与线程锁照搬 c6_runs.py（防 torch segfault）。

用法：
  python p1_5_ablation.py --workers 6 --seeds 42        # 先跑 seed42 定位
  python p1_5_ablation.py --workers 6 --seeds 42,123,777  # 补多种子显著性
增量写 p1_5_runs.json；已完成的任务自动跳过。
"""
import os
import json
import time
import argparse
import subprocess
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"C:\Users\Lenovo\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
MASTER = os.path.join(HERE, "p1_5_runs.json")

THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "TORCH_NUM_THREADS": "1",
}


def build_jobs(bs, lr, epochs, seeds, hist_mode="user", seq_len=200):
    common = ["--device", "cuda", "--max_degree", "10",
              "--batch_size", str(bs), "--lr", str(lr), "--epochs", str(epochs),
              "--bge_model_dir", "bge_model", "--bge_cache", "poi_bge_emb.npy",
              "--use_bge", "--sem_thr", "0.90",
              "--scorer", "dot", "--session_pool", "mean", "--use_sgcp", "--ours_only",
              "--hist_mode", hist_mode, "--seq_len", str(seq_len),
              "--prior_channels", "cnt,rec,pop", "--gate_mode", "context"]
    jobs = []

    # ---- 单边缘类型消融（4 类边各 only）----
    # geo-only：关 cat/sem/covisit，保留 geo（需数据集有 geo → Foursquare 满足）
    for s in seeds:
        jobs.append((f"geo_only_s{s}", common +
                     ["--no_category_edges", "--no_semantic_edges", "--no_covisit_edges", "--seed", str(s)]))
    # cat-only：关 geo/sem/covisit，保留 category
    for s in seeds:
        jobs.append((f"cat_only_s{s}", common +
                     ["--no_geo_edges", "--no_semantic_edges", "--no_covisit_edges", "--seed", str(s)]))
    # sem-only：关 geo/cat/covisit，保留 semantic（需 use_bge）
    for s in seeds:
        jobs.append((f"sem_only_s{s}", common +
                     ["--no_geo_edges", "--no_category_edges", "--no_covisit_edges", "--seed", str(s)]))
    # covisit-only：关 geo/cat/sem，保留纯行为共访边
    for s in seeds:
        jobs.append((f"covisit_only_s{s}", common +
                     ["--no_geo_edges", "--no_category_edges", "--no_semantic_edges", "--seed", str(s)]))

    # ---- 传播深度消融（L=1 vs L=2，全部四类边）----
    for s in seeds:
        jobs.append((f"depth1_s{s}", common +
                     ["--num_gnn_layers", "1", "--seed", str(s)]))
    for s in seeds:
        jobs.append((f"depth2_s{s}", common +
                     ["--num_gnn_layers", "2", "--seed", str(s)]))
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
                          ("num_gnn_layers", "use_geo_edges", "use_category_edges",
                           "use_semantic_edges", "use_covisit_edges", "prior_channels",
                           "gate_mode", "use_kg_channel", "no_graph")}
            rec["protocol"] = {k: td.get(k) for k in
                               ("hist_mode", "seq_len", "n_samples",
                                "hist_len_mean", "revisit_ratio")}
            rec["train_seconds"] = td.get("train_seconds")
            cs = d.get("cold_start(\u22645)", {})
            rec["cold_n"] = cs.get("n")
            rec["ours_cold"] = cs.get("results", {}).get("LLM-STKG (ours)")
            diag = d.get("rank_diag", {}).get("full", {}).get("LLM-STKG (ours)", {})
            rec["rank_diag"] = {k: v for k, v in diag.items() if k != "ranks"}
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
    ap.add_argument("--seeds", default="42", help="逗号分隔种子列表，默认仅 42（定位用）")
    ap.add_argument("--only", default=None, help="逗号分隔，仅跑指定任务名")
    ap.add_argument("--hist_mode", default="user", choices=["trajectory", "user"])
    ap.add_argument("--seq_len", type=int, default=200)
    a = ap.parse_args()

    from _singleton import acquire
    acquire("_p1_5_driver.lock", "p1_5_ablation.py")
    MASTER = os.path.join(HERE, "p1_5_runs.json")

    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    jobs = build_jobs(a.bs, a.lr, a.epochs, seeds, a.hist_mode, a.seq_len)
    if a.only:
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if j[0] in keep]
    jobs = [(n, e) for n, e in jobs]

    m = load_master()
    m["config"] = {"bs": a.bs, "lr": a.lr, "epochs": a.epochs, "max_degree": 10,
                   "seeds": seeds, "hist_mode": a.hist_mode, "seq_len": a.seq_len,
                   "note": "P1-5 single-edge-type + propagation-depth ablation; "
                           "production NYC protocol (use_bge/use_sgcp/C6 context)",
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
            ours = rec.get("ours", {})
            print(f"[{ts()}] DONE {name} rc={p.returncode} "
                  f"R@5={ours.get('R@5') if ours else None} "
                  f"R@10={ours.get('R@10') if ours else None} "
                  f"N@10={ours.get('NDCG@10') if ours else None}", flush=True)
        running = still


if __name__ == "__main__":
    main()
