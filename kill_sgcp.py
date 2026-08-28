"""精确终止 sgcp_sweep 驱动器及其 head_to_head 子进程，释放 CPU 给 NDCG 诊断。
注意：仅在诊断尚未启动时运行（此时所有 head_to_head 都是 sgcp_sweep 的子进程）。
"""
import psutil, subprocess

def kill(pid):
    try:
        r = subprocess.run(['taskkill', '/PID', str(pid), '/F', '/T'],
                            capture_output=True, text=True)
        return r.stdout.strip() or r.stderr.strip()
    except Exception as e:
        return f"err {e}"

print("=== drivers (sgcp_sweep.py) ===")
for p in psutil.process_iter(['pid', 'name', 'cmdline']):
    if p.info['name'] != 'python.exe':
        continue
    cl = ' '.join(p.info['cmdline'] or [])
    if 'sgcp_sweep' in cl:
        print(kill(p.info['pid']), '|', cl[:60])

print("=== orphaned head_to_head children ===")
for p in psutil.process_iter(['pid', 'name', 'cmdline']):
    if p.info['name'] != 'python.exe':
        continue
    cl = ' '.join(p.info['cmdline'] or [])
    if 'head_to_head' in cl:
        print(kill(p.info['pid']), '|', cl[:60])

print("=== remaining check ===")
left = [(p.info['pid'], ' '.join(p.info['cmdline'] or [])[:50])
       for p in psutil.process_iter(['pid', 'name', 'cmdline'])
       if p.info['name'] == 'python.exe'
       and ('sgcp_sweep' in ' '.join(p.info['cmdline'] or [])
            or 'head_to_head' in ' '.join(p.info['cmdline'] or []))]
print('remaining:', left if left else 'NONE')
