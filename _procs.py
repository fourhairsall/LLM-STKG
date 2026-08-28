"""正确统计实验进程数（去掉 venv python 启动器 shim 造成的重复计数）。

坑：Windows 上 venv 的 `Scripts\\python.exe` 是启动器 shim，会再 exec 一个
命令行完全相同的子进程。用 psutil 直接按 cmdline 匹配会把每个逻辑进程数成
两份，极易被误判为"重复启动"而错杀正常任务。

判定规则：若某进程的父进程 cmdline 与自身相同，则它是 shim 的子体，不单独计数
（只保留最外层那个作为逻辑进程）。

用法：python _procs.py [关键字...]（默认 p0_pilot.py / p0_runs.py / head_to_head）
"""
import os
import sys
import datetime

import psutil

DEFAULT_KEYS = ["p0_pilot.py", "p0_runs.py", "llm_stkg.head_to_head"]


def logical_procs(keys=None):
    keys = keys or DEFAULT_KEYS
    me = os.getpid()
    cand = {}
    for p in psutil.process_iter(["pid", "ppid", "name", "cmdline", "create_time"]):
        try:
            cl = p.info["cmdline"] or []
            if p.info["pid"] == me or ("-c" in cl[:2]):
                continue
            if "python" not in (p.info["name"] or "").lower():
                continue
            s = " ".join(cl)
            if any(k in s for k in keys):
                cand[p.info["pid"]] = (p.info["ppid"], s, p.info["create_time"])
        except Exception:
            pass
    out = []
    for pid, (ppid, s, ct) in cand.items():
        parent = cand.get(ppid)
        if parent is not None and parent[1] == s:
            continue  # 自己是 shim 的子体，父进程已计数
        out.append((pid, s, ct))
    return sorted(out, key=lambda x: x[2])


if __name__ == "__main__":
    ks = sys.argv[1:] or None
    rows = logical_procs(ks)
    for pid, s, ct in rows:
        t = datetime.datetime.fromtimestamp(ct).strftime("%H:%M:%S")
        print(f"{pid:>7}  {t}  {s[-95:]}")
    print(f"LOGICAL_COUNT = {len(rows)}")
    print(f"CPU={psutil.cpu_percent(interval=1)}%  MEM={psutil.virtual_memory().percent}%")
