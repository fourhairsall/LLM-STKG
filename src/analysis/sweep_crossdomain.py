"""跨域配置定位扫描：找出 LM-STKG 在无重复消费域（Steam/MovieLens）不收敛的根因。

背景
----
Foursquare-NYC 上训练 loss 从 1.2522 一路降到 0.0590（30 轮），但换到 Steam 后
loss 从第 2 轮起死锁在 2.2096~2.2101（neg=10 的 11 类 CE 随机基准 ln(11)=2.3979），
评测端 R@10 甚至低于零参数 Popularity。已排除的因素：
  1. 评测未屏蔽历史 —— 已修（evaluate.mask_history，按 revisit_ratio 自动开关）；
  2. cnt/rec 通道在 revisit≈0 域恒零且非负门控无法反转 —— 已修（自动剔除）；
  3. 打分器/池化沿用了已弃用的 mlp/gru 默认 —— 已修（对齐生产配置 dot/mean）。
上述三项修完后 loss 仍锁死，故本脚本对「会话编码器 × 学习率 × 训练轮数」做定位扫描。

主要假设
--------
H1（池化）：hist_mode=user 下 Steam 历史均长 53.6、MovieLens 174.5，对如此长的历史做
           **均值池化**会把用户向量压成近似全局均值，判别信息被抹平。Foursquare 早期
           选定 mean 时用的是 trajectory 模式（历史均长 7.8），结论不可直接外推。
H2（学习率）：lr=4e-3 是在 Foursquare「cnt 通道提供强捷径」的条件下调出来的；无捷径域
           可能过大导致早期塌到平凡解后梯度消失。
H3（轮数）：8 轮 × 184 步/轮 = 1472 步，可能纯粹欠训练。

用法
----
  python sweep_crossdomain.py --dataset steam --epochs 6
  python sweep_crossdomain.py --dataset steam --epochs 6 --arms dot_mean_4e3,dot_gru_4e3
输出：sweep_crossdomain_<dataset>.json + 终端汇总表（末轮 loss / R@10 / 对照零参数水位）
"""
import argparse
import json
import os
import re
import subprocess
import sys

PY = sys.executable
ENV_PREFIX = {
    # 铁律：OpenMP 与 OpenBLAS 线程混载会导致 torch 偶发 segfault，六个变量缺一不可
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "TORCH_NUM_THREADS": "1",
}

# 各域零参数水位（quick_domain_probe.py 实测，已修正标签泄漏），用于判断"是否真的学到东西"
FLOOR = {
    "steam":   {"protocol": "mask_hist=ON  (revisit 0.0000)", "Random": 0.0212, "Pop": 0.0725, "ItemKNN": 0.0809},
    "ml1m":    {"protocol": "mask_hist=ON  (revisit 0.0000)", "Random": 0.0175, "Pop": 0.0315, "ItemKNN": 0.0330},
    "gowalla": {"protocol": "mask_hist=OFF (revisit 0.6759)", "Random": 0.0056, "Pop": 0.0572, "ItemKNN": 0.2161},
}

ARMS = {
    # name            : (scorer, session_pool, lr)
    "dot_mean_4e3":   ("dot", "mean", "4e-3"),   # 当前跨域默认（= Foursquare 生产配置）
    "dot_gru_4e3":    ("dot", "gru",  "4e-3"),   # H1：换回序列编码器
    "dot_gru_1e3":    ("dot", "gru",  "1e-3"),   # H1+H2
    "dot_mean_1e3":   ("dot", "mean", "1e-3"),   # H2 单独
    "mlp_gru_1e3":    ("mlp", "gru",  "1e-3"),   # 参照：早期默认路径
}

SIZES = {  # pilot 规模（POI 数, 用户数）
    "steam":   (600, 2500),
    "ml1m":    (800, 2000),
    "gowalla": (1500, 3000),
}


def run_arm(name, dataset, epochs, batch_size):
    scorer, pool, lr = ARMS[name]
    mp, mu = SIZES[dataset]
    out_json = f"sw_{dataset}_{name}.json"
    log = f"_sw_{dataset}_{name}.log"
    cmd = [PY, "-u", "-m", "llm_stkg.head_to_head",
           "--device", "cuda", "--dataset", dataset,
           "--ds_max_pois", str(mp), "--ds_max_users", str(mu),
           "--batch_size", str(batch_size), "--lr", lr, "--epochs", str(epochs),
           "--scorer", scorer, "--session_pool", pool,
           "--ours_only", "--out", out_json]
    env = dict(os.environ); env.update(ENV_PREFIX)
    print(f"\n>>> [{name}] scorer={scorer} pool={pool} lr={lr} epochs={epochs}", flush=True)
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    txt = open(log, encoding="utf-8", errors="replace").read()
    losses = [float(x) for x in re.findall(r"\[ours Epoch \d+\] loss=([\d.]+)", txt)]
    rec = {"arm": name, "scorer": scorer, "pool": pool, "lr": lr,
           "returncode": p.returncode,
           "loss_first": losses[0] if losses else None,
           "loss_last": losses[-1] if losses else None,
           "loss_curve": losses}
    if os.path.exists(out_json):
        try:
            d = json.load(open(out_json, encoding="utf-8"))
            o = d.get("results", {}).get("LLM-STKG (ours)", {})
            rec.update({"Recall@5": o.get("Recall@5"), "Recall@10": o.get("Recall@10"),
                        "NDCG@10": o.get("NDCG@10"),
                        "mask_history": d.get("mask_history"),
                        "revisit_ratio_test": d.get("revisit_ratio_test")})
        except Exception as e:                                   # noqa: BLE001
            rec["parse_error"] = str(e)
    print(f"    loss {rec['loss_first']} -> {rec['loss_last']} | "
          f"R@10={rec.get('Recall@10')}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="steam", choices=list(SIZES))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a in ARMS]
    recs = [run_arm(a, args.dataset, args.epochs, args.batch_size) for a in arms]

    fl = FLOOR.get(args.dataset, {})
    print("\n" + "=" * 78)
    print(f"跨域配置扫描汇总 — {args.dataset}   协议: {fl.get('protocol', 'n/a')}")
    print("=" * 78)
    print(f"{'arm':<16}{'scorer':<8}{'pool':<7}{'lr':<8}{'loss首':<9}{'loss末':<9}{'R@10':<9}{'N@10':<9}")
    print("-" * 78)
    for r in recs:
        print(f"{r['arm']:<16}{r['scorer']:<8}{r['pool']:<7}{r['lr']:<8}"
              f"{r['loss_first'] if r['loss_first'] is not None else '-':<9}"
              f"{r['loss_last'] if r['loss_last'] is not None else '-':<9}"
              f"{r.get('Recall@10', '-') if r.get('Recall@10') is not None else '-':<9}"
              f"{r.get('NDCG@10', '-') if r.get('NDCG@10') is not None else '-':<9}")
    print("-" * 78)
    print(f"{'零参数水位':<16}{'':<8}{'':<7}{'':<8}{'':<9}{'':<9}"
          f"Random={fl.get('Random')} Pop={fl.get('Pop')} ItemKNN={fl.get('ItemKNN')}")
    print("判读：R@10 若不显著高于 ItemKNN-cooc，则模型未学到超越共现记忆的结构。")

    out = f"sweep_crossdomain_{args.dataset}.json"
    json.dump({"dataset": args.dataset, "epochs": args.epochs,
               "zero_param_floor": fl, "arms": recs},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
