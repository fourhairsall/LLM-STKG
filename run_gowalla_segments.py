"""
Gowalla 7B-LoRA segmented driver.
Background python processes here are silently reclaimed after ~5 min, so we chain
short segments (max_gsteps=120 ~= 3 min) each saving a checkpoint, then a final
segment that resumes the last checkpoint AND runs the full eval (<1 min, safe).
"""
import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "open_llama_7b_v2")
SEG = 30           # optimizer steps per segment (~2.5 min, safe before ~5-min reclaim)
N_SEG = 34         # 34*30 = 1020 steps ~= full 3-epoch training for Gowalla
OUT = "llm4poi_openllama7b_gowalla.json"
common = ["--peft", "lora", "--model_dir", MODEL, "--load_in_4bit",
          "--batch", "2", "--grad_accum", "8", "--lr", "1e-4",
          "--no_grad_ckpt", "--data_root", "../data/gowalla/processed",
          "--max_gsteps", str(SEG), "--save_every", str(SEG)]

def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    print("<< exit", r.returncode, flush=True)
    return r.returncode

for seg in range(1, N_SEG):
    resume = None
    if seg > 1:
        resume = os.path.join(HERE, f"llm4poi_{os.path.basename(MODEL)}_gowalla_lora_ckpt{seg*SEG}")
    cmd = [sys.executable, "llm4poi_baseline_ptuning.py", "--city", "gowalla",
           "--out", f"/tmp/gowalla_seg{seg}.json"] + common
    if resume and os.path.isdir(resume):
        cmd += ["--resume_adapter", resume]
    run(cmd)

# final segment: resume last ckpt, run full eval
last_ckpt = os.path.join(HERE, f"llm4poi_{os.path.basename(MODEL)}_gowalla_lora_ckpt{(N_SEG-1)*SEG}")
cmd = [sys.executable, "llm4poi_baseline_ptuning.py", "--city", "gowalla",
       "--out", OUT] + common
if os.path.isdir(last_ckpt):
    cmd += ["--resume_adapter", last_ckpt]
run(cmd)
print("ALL GOWALLA SEGMENTS DONE", flush=True)
