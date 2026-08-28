"""单实例锁。

背景：Windows + Git Bash 后台启动（nohup / 工具级 background）偶发把同一条驱动
命令执行两次，导致两个驱动 + 双倍子进程抢 CPU，结果互相拖慢且日志互相覆盖。
在驱动脚本入口调用 acquire() 即可从根上杜绝双开。
"""
import os
import sys
import atexit

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))


def _ancestors(pid: int) -> set:
    """本进程及其所有祖先。

    Windows 上 venv 的 Scripts\\python.exe 是 launcher shim，它会再拉起一个真正的
    解释器进程，两者 cmdline 相同。扫描时必须把自己这条链整体排除，否则会把自己的
    父 shim 误判成"已有实例"。
    """
    out, cur, depth = set(), pid, 0
    while cur and depth < 8:
        out.add(cur)
        try:
            cur = psutil.Process(cur).ppid()
        except Exception:
            break
        depth += 1
    return out


def _scan_live(token: str, exclude: set):
    """全表扫描 cmdline 含 token 的活进程。返回 pid 或 None。"""
    for p in psutil.process_iter(["pid", "cmdline"]):
        pid = p.info.get("pid")
        if pid in exclude:
            continue
        try:
            cl = " ".join(p.info.get("cmdline") or [])
        except Exception:
            continue
        if token in cl:
            return pid
    return None


def acquire(lock_name: str, token: str) -> str:
    """若已有同名活实例在跑则直接退出；否则写入本进程 PID 并注册退出清理。

    lock_name: 锁文件名，如 "_p0_pilot.lock"
    token:     用于二次确认的命令行特征串，如 "p0_pilot.py"

    2026-08-01 修复：原实现只信任锁文件里记录的 PID。但在 Windows venv 下写锁的可能
    是 launcher shim，shim 退出后 PID 立刻失效，锁形同虚设 —— 实测因此双开过一次
    （24 个子进程互抢 CPU）。现改为「锁文件 + 全表 cmdline 扫描」双保险。
    """
    lock = os.path.join(HERE, lock_name)
    me = _ancestors(os.getpid())

    # 主互斥 = O_EXCL 原子创建。绝不能用「全表扫描 cmdline」做主判据：两份几乎同时
    # 启动时会互相扫到对方，然后双双退出（2026-08-01 实测踩过）。扫描只在锁陈旧时
    # 作为二次确认。
    for attempt in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            old = -1
            try:
                old = int(open(lock, encoding="utf-8").read().strip())
            except Exception:
                old = -1
            alive = old > 0 and old not in me and psutil.pid_exists(old)
            if alive:
                try:
                    cl = " ".join(psutil.Process(old).cmdline() or [])
                except Exception:
                    cl = ""
                if token in cl:
                    print(f"[singleton] 已有实例在运行 pid={old}，本次启动直接退出。",
                          flush=True)
                    sys.exit(0)
            # 锁里的 PID 已死或不是本程序：再全表确认一次，确实没活实例才夺锁
            live = _scan_live(token, me)
            if live is not None:
                print(f"[singleton] 已有实例在运行 pid={live}（扫描命中），本次启动直接退出。",
                      flush=True)
                sys.exit(0)
            if attempt == 0:
                try:
                    os.remove(lock)
                except Exception:
                    pass
                continue
            with open(lock, "w", encoding="utf-8") as f:
                f.write(str(os.getpid()))

    def _cleanup():
        try:
            if os.path.exists(lock):
                os.remove(lock)
        except Exception:
            pass

    atexit.register(_cleanup)
    return lock
