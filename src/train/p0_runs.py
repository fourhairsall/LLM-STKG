"""P0 全量重跑驱动器（[2,E] 边索引 bug 修复 + k-NN 剪枝之后）。

背景：stkg_net 此前把 kg_builder 输出的 [2,E] 边索引又转置了一次，导致
`src, dst = ei[0], ei[1]` 只取到前两条边——异构图传播实际近乎空转。修复后
图真正参与传播，此前所有 ours 侧数值全部作废，需要按同一协议重跑。

用法（后台）：
  python p0_runs.py --bs 256 --lr 2e-3 --epochs 30 --workers 8
增量写 p0_runs.json；每个任务独立 .log / .json，进程被杀后可直接续跑（已完成的跳过）。
"""
import os
import sys
import json
import time
import argparse
import subprocess
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"C:\Users\Lenovo\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
MASTER = os.path.join(HERE, "p0_runs.json")

THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "TORCH_NUM_THREADS": "1",
}

SEEDS = [42, 123, 777, 2024, 99]


def build_jobs(bs, lr, epochs):
    """返回 [(name, extra_args, is_full)]。"""
    common = ["--device", "cpu", "--max_degree", "10",
              "--batch_size", str(bs), "--lr", str(lr), "--epochs", str(epochs),
              "--bge_model_dir", "bge_model", "--bge_cache", "poi_bge_emb.npy"]
    bge = ["--use_bge", "--sem_thr", "0.90"]
    head = ["--scorer", "dot", "--session_pool", "mean"]
    main_cfg = common + bge + head + ["--use_sgcp"]
    jobs = []

    # ---- 三段消融（C1 / SGCP）----
    jobs.append(("abl_base", common + head + ["--seed", "42", "--ours_only"], False))
    jobs.append(("abl_c1", common + bge + head + ["--seed", "42", "--ours_only"], False))

    # ---- C1 贡献拆分（诚信要求）----
    # 本数据集的 POI "文本描述" 由 f"{cat_name} near {lat:.2f},{lng:.2f}" 合成，
    # 其信息上界 = 类目名 + ~1.1km 网格坐标，并非真实世界的 POI 描述文本。
    # 因此必须把 C1 的收益拆开报告，否则会让读者高估"语言模型语义知识"的作用：
    #   c1_featonly : 有 BGE 节点特征，无语义边（sem_thr=1.01 → 语义边集合为空）
    #   c1_edgeonly : 无 BGE 节点特征（sem_feat_mode=none），保留语义边 + SGCP
    #   c1_catleak  : 用类目 one-hot 替代 BGE 作节点特征 —— 若与完整版相当，
    #                 说明"语义"增益主要来自类目先验（类目名泄漏），须如实披露
    jobs.append(("abl_c1_featonly", common + ["--use_bge", "--sem_thr", "1.01"] + head
                 + ["--use_sgcp", "--seed", "42", "--ours_only"], False))
    jobs.append(("abl_c1_edgeonly", main_cfg + ["--sem_feat_mode", "none",
                                                "--seed", "42", "--ours_only"], False))
    jobs.append(("abl_c1_catleak", main_cfg + ["--sem_feat_mode", "cat_onehot",
                                               "--seed", "42", "--ours_only"], False))

    # ---- C2 / C3 / 残差 消融（审稿人 A、F、G 的 P0 要求）----
    jobs.append(("abl_homo", main_cfg + ["--homo_gnn", "--seed", "42", "--ours_only"], False))
    jobs.append(("abl_gru", common + bge + ["--use_sgcp", "--scorer", "dot",
                                            "--session_pool", "gru",
                                            "--seed", "42", "--ours_only"], False))
    jobs.append(("abl_nores", main_cfg + ["--no_residual", "--seed", "42", "--ours_only"], False))

    # ---- 语义阈值 τ 敏感度（审稿人 B：τ 疑似循环调参）----
    for tau in ("0.85", "0.95"):
        args = [a for a in main_cfg]
        args[args.index("--sem_thr") + 1] = tau
        jobs.append((f"tau{tau.replace('.', '')}", args + ["--seed", "42", "--ours_only"], False))

    # ---- 训练目标对照（审稿人 D：失败目标需给全指标面板）----
    jobs.append(("obj_bpr", main_cfg + ["--loss", "bpr", "--seed", "42", "--ours_only"], False))
    jobs.append(("obj_list", main_cfg + ["--loss", "list", "--tau", "0.5",
                                         "--seed", "42", "--ours_only"], False))
    jobs.append(("obj_hardneg", main_cfg + ["--hard_neg_ratio", "0.5",
                                            "--seed", "42", "--ours_only"], False))
    jobs.append(("obj_neg100", main_cfg + ["--neg_samples", "100",
                                           "--seed", "42", "--ours_only"], False))

    # ---- 5 种子全量（含全部基线，供配对显著性检验）----
    # seed 42 额外保存权重，供 explain.py 生成真实 KG 路径解释（§4.5/§5.8），免重训
    for s in SEEDS:
        extra = main_cfg + ["--seed", str(s)]
        if s == 42:
            extra = extra + ["--save_model", os.path.join(HERE, "_best_seed42.pt")]
        jobs.append((f"sig_seed{s}", extra, True))
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
            r = d.get("results", {})
            rec["ours"] = r.get("LLM-STKG (ours)")
            for b in ("LightGCN", "BPR-MF", "FPMC", "GRU-STGN", "Popularity (Pop)"):
                if b in r:
                    rec.setdefault("baselines", {})[b] = r[b]
            rec["train_diag"] = d.get("train_diag")
            cs = d.get("cold_start(\u22645)", {})
            rec["cold_n"] = cs.get("n")
            rec["ours_cold"] = cs.get("results", {}).get("LLM-STKG (ours)")
            if cs.get("results", {}).get("LightGCN"):
                rec["lightgcn_cold"] = cs["results"]["LightGCN"]
            diag = d.get("rank_diag", {}).get("full", {}).get("LLM-STKG (ours)", {})
            rec["rank_diag"] = {k: v for k, v in diag.items() if k != "ranks"}
        except Exception as e:
            rec["status"] = f"json_err:{e}"
    return rec


def main():
    from _singleton import acquire
    acquire("_p0_runs.lock", "p0_runs.py")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", default="2e-3")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None, help="逗号分隔，仅跑指定任务名")
    a = ap.parse_args()

    jobs = build_jobs(a.bs, a.lr, a.epochs)
    if a.only:
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if j[0] in keep]

    m = load_master()
    m["config"] = {"bs": a.bs, "lr": a.lr, "epochs": a.epochs, "max_degree": 10,
                   "note": "post edge-index fix + kNN pruning", "started": ts()}
    m.setdefault("jobs", {})
    pending = [j for j in jobs if m["jobs"].get(j[0], {}).get("status") != "ok"]
    print(f"[{ts()}] total={len(jobs)} pending={len(pending)} workers={a.workers}", flush=True)
    save(m)

    running = []
    queue = list(pending)
    while queue or running:
        while queue and len(running) < a.workers:
            name, extra, is_full = queue.pop(0)
            out_json = os.path.join(HERE, f"p0_{name}.json")
            log = os.path.join(HERE, f"p0_{name}.log")
            cmd = [PY, "-m", "llm_stkg.head_to_head"] + extra + ["--out", out_json]
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
                  f"R@5={o.get('Recall@5')} R@10={o.get('Recall@10')} N@10={o.get('NDCG@10')}",
                  flush=True)
        running = still
    m["finished"] = ts()
    save(m)
    print(f"[{ts()}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
