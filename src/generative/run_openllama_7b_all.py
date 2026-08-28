"""
Driver: run OpenLLaMA-7B-v2 + P-tuning v2 generative baseline on NYC, TKY, Gowalla
sequentially on a single 12 GB GPU (must serialise -- no concurrent 7B loads).

Protocol matched to the paper's 3B Qwen+LoRA baseline:
  peft=ptuning (P-tuning v2, prefix=20), load_in_8bit (embed/lm_head kept fp16
  so BGE seeding survives), batch=2, grad_accum=8 (effective 16 == 3B's 4x4),
  epochs=3, lr=1e-3, no_grad_ckpt (P-tuning incompatible with grad-ckpt on
  transformers>=5.x).

Each city writes its own JSON (llm4poi_openllama7b_<city>.json) + adapter, so a
failure on one city does not lose the others.
"""
import sys, os, traceback, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import llm4poi_baseline_ptuning as M

MODEL = os.path.join(HERE, "models", "open_llama_7b_v2")
cities = ["nyc", "tky", "gowalla"]
common = ["--peft", "lora", "--model_dir", MODEL, "--load_in_4bit",
          "--batch", "2", "--grad_accum", "8", "--epochs", "3", "--lr", "1e-4", "--no_grad_ckpt"]

results = {}
t_all = time.time()
for c in cities:
    out = os.path.join(HERE, f"llm4poi_openllama7b_{c}.json")
    argv = ["prog", "--city", c] + common + ["--out", out]
    sys.argv = argv
    print(f"\n##### CITY={c}  start ######", flush=True)
    t0 = time.time()
    try:
        M.main()
        results[c] = f"OK ({time.time()-t0:.1f}s) -> {out}"
    except Exception as e:
        results[c] = f"FAIL: {type(e).__name__}: {e}"
        traceback.print_exc()
    # free GPU before next city
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"##### CITY={c}  done: {results[c]} #####\n", flush=True)

print("=== ALL CITIES DONE (total %.1f min) ===" % ((time.time()-t_all)/60), flush=True)
for k, v in results.items():
    print(f"  {k}: {v}", flush=True)
