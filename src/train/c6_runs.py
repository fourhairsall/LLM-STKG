"""C6（行为先验通道 + 上下文门控）全量实验驱动器。

动机（诚信修订 P0）：本基准 75.7% 的测试目标已出现在该用户自身历史中，零参数的
History-Frequency 规则 R@10=0.6275，高于我们此前所有神经模型（ours 0.4755）。
C6 把「历史计数 / 近因 / 全局热度」三类行为先验显式接入打分头，并用会话上下文
学门控权重，使打分函数的假设空间严格包含 HF / HR / Pop —— 因此不可能系统性劣于
平凡基线。

本驱动器要回答的四个问题（每个都对应一条审稿意见）：
  Q1 C6 完整版能否稳定超过 HF(0.6275)？        → c6_full_s{5 seeds}
  Q2 去掉整条 KG 语义通道后还剩多少？          → c6_nokg_s{3 seeds}  ★致命消融
  Q3 上下文门控相对全局标量权重的净贡献？      → c6_global_s42
  Q4 三个先验通道各自的边际贡献？              → c6_cnt / c6_cntrec / c6_off

用法（后台）：
  python c6_runs.py --workers 8
增量写 c6_runs.json；每个任务独立 .log / .json，进程被杀后可直接续跑（已完成的跳过）。
"""
import os
import json
import time
import argparse
import subprocess
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"C:\Users\Lenovo\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
MASTER = os.path.join(HERE, "c6_runs.json")   # main() 中按 --tag 覆盖

# 线程全锁 1：torch 与 OpenBLAS 混载会偶发 segfault，这是本机的既定铁律
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "TORCH_NUM_THREADS": "1",
}

SEEDS = [42, 123, 777, 2024, 99]
ABL_SEEDS = [42, 123, 777]     # 关键消融也要多种子，否则无法排除单次波动


def build_jobs(bs, lr, epochs, hist_mode="user", seq_len=200):
    """返回 [(name, extra_args)]。所有任务均 --ours_only（基线排名由 baseline_ranks.json 提供）。

    hist_mode / seq_len 是 2026-08-01 定位到的训练-测试协议错配的修复项：
    旧默认 trajectory/20 让训练样本的历史均值只有 7.8 步、重访率 38.6%，而官方测试
    给的是用户跨会话全历史（143.2 步 / 75.7%）。user/200 把训练重访率对齐到 0.7588,
    与测试端 0.7574 基本一致。任何依赖历史统计的通道（cnt/rec）在旧协议下都被系统
    性低估，因此本轮全部任务必须在新协议下重跑。
    """
    common = ["--device", "cuda", "--max_degree", "10",
              "--batch_size", str(bs), "--lr", str(lr), "--epochs", str(epochs),
              "--bge_model_dir", "bge_model", "--bge_cache", "poi_bge_emb.npy",
              "--use_bge", "--sem_thr", "0.90",
              "--scorer", "dot", "--session_pool", "mean", "--use_sgcp", "--ours_only",
              "--hist_mode", hist_mode, "--seq_len", str(seq_len)]
    c6 = ["--prior_channels", "cnt,rec,pop", "--gate_mode", "context"]
    jobs = []

    # ---- Q1 完整 C6，5 种子（主结果 + 显著性）----
    for s in SEEDS:
        extra = common + c6 + ["--seed", str(s)]
        if s == 42:
            # seed 42 存权重：explain.py 用它生成真实 KG 路径解释（§5.8），免重训
            extra = extra + ["--save_model", os.path.join(HERE, "_c6u_seed42.pt")]
        jobs.append((f"c6_full_s{s}", extra))

    # ---- Q2 ★致命消融：整条 KG 语义通道移除（不参与 stack，非权重置零）----
    for s in ABL_SEEDS:
        jobs.append((f"c6_nokg_s{s}", common + c6 + ["--no_kg_channel", "--seed", str(s)]))

    # ---- Q3 门控形式消融：全局标量 vs 上下文相关 ----
    for s in ABL_SEEDS:
        jobs.append((f"c6_global_s{s}", common + ["--prior_channels", "cnt,rec,pop",
                                                  "--gate_mode", "global", "--seed", str(s)]))

    # ---- Q4 通道边际贡献（seed 42）----
    jobs.append(("c6_cnt_s42", common + ["--prior_channels", "cnt",
                                         "--gate_mode", "context", "--seed", "42"]))
    jobs.append(("c6_cntrec_s42", common + ["--prior_channels", "cnt,rec",
                                            "--gate_mode", "context", "--seed", "42"]))
    jobs.append(("c6_pop_s42", common + ["--prior_channels", "pop",
                                         "--gate_mode", "context", "--seed", "42"]))
    # 无 C6 对照（= 修订前的 ours 主模型，同一 bs/lr/epochs 协议下重跑，保证单变量）
    for s in ABL_SEEDS:
        jobs.append((f"c6_off_s{s}", common + ["--seed", str(s)]))

    # ---- Q5 ★C2 净贡献：完全不做异构消息传递（只保留 skip 线性投影）----
    # 特征编码器 / 维度 / 打分头 / 训练目标全不变，唯一变量是有无图传播。
    for s in ABL_SEEDS:
        jobs.append((f"nograph_s{s}", common + c6 + ["--no_graph", "--seed", str(s)]))
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
            # 只留与 C6 相关的诊断，主 json 保持可读
            rec["c6"] = {k: td.get(k) for k in
                         ("prior_channels", "gate_mode", "use_kg_channel", "no_graph",
                          "gate_w_mean")}
            # 协议指纹：确保表格里每一行都能追溯到它是在哪套 train/test 协议下产生的
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
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None, help="逗号分隔，仅跑指定任务名")
    ap.add_argument("--hist_mode", default="user", choices=["trajectory", "user"])
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--tag", default="u", help="输出文件前缀，隔离不同协议的批次")
    a = ap.parse_args()

    from _singleton import acquire
    # token 必须是真实 cmdline 的子串才能被全表扫描命中。同一时刻只允许一个 c6_runs
    # 驱动器（不同 tag 也不该并存，否则 16 核会被两批任务对半劈）。
    acquire("_c6_driver.lock", "c6_runs.py")
    MASTER = os.path.join(HERE, f"{a.tag}_runs.json")

    jobs = build_jobs(a.bs, a.lr, a.epochs, a.hist_mode, a.seq_len)
    if a.only:
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if j[0] in keep]
    # 前缀在此处统一加上，保证 pending 判定与落盘用的是同一个 key（否则永远判为未完成）
    jobs = [(f"{a.tag}_{n}", e) for n, e in jobs]

    m = load_master()
    m["config"] = {"bs": a.bs, "lr": a.lr, "epochs": a.epochs, "max_degree": 10,
                   "hist_mode": a.hist_mode, "seq_len": a.seq_len,
                   "note": "C6 prior channels + context gate; train/test protocol aligned",
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
            # -u：子进程 stdout 非 TTY 时会被全缓冲，日志在进程结束前一直是 0 字节，
            # 无法在线观察进度。加 -u 强制行缓冲。
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
                  f"gate_w={(rec.get('c6') or {}).get('gate_w_mean')}", flush=True)
        running = still
    m["finished"] = ts()
    save(m)
    print(f"[{ts()}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
