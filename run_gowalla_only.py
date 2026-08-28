import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "open_llama_7b_v2")
common = ["--peft", "lora", "--model_dir", MODEL, "--load_in_4bit",
          "--batch", "2", "--grad_accum", "8", "--epochs", "3", "--lr", "1e-4",
          "--no_grad_ckpt", "--data_root", "../data/gowalla/processed"]
cmd = [sys.executable, "llm4poi_baseline_ptuning.py", "--city", "gowalla",
       "--out", "llm4poi_openllama7b_gowalla.json"] + common
print("CITY=gowalla cmd=", " ".join(cmd), flush=True)
r = subprocess.run(cmd, cwd=HERE)
print("gowalla EXIT=", r.returncode, flush=True)
print("ALL DONE", flush=True)
