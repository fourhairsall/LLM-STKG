import os, sys, time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from huggingface_hub import snapshot_download

REPO = "openlm-research/open_llama_7b_v2"
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.join(HERE, "models", "open_llama_7b_v2")

t0 = time.time()
print(f"[download] repo={REPO} -> {LOCAL}", flush=True)
path = snapshot_download(
    repo_id=REPO,
    local_dir=LOCAL,
    local_dir_use_symlinks=False,
)
dt = time.time() - t0
print(f"[download] DONE in {dt/60:.1f} min -> {path}", flush=True)
# report size
tot = 0
for root, _, files in os.walk(LOCAL):
    for f in files:
        tot += os.path.getsize(os.path.join(root, f))
print(f"[download] total bytes={tot} ({tot/1e9:.2f} GB)", flush=True)
