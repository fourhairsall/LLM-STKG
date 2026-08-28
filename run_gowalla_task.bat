@echo off
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set VECLIB_MAXIMUM_THREADS=1
set TORCH_NUM_THREADS=1
"C:\Users\Lenovo\.workbuddy\binaries\python\envs\default\Scripts\python.exe" llm4poi_baseline_ptuning.py --city gowalla --peft lora --model_dir models/open_llama_7b_v2 --load_in_4bit --batch 2 --grad_accum 8 --epochs 3 --lr 1e-4 --no_grad_ckpt --data_root "../data/gowalla/processed" --out llm4poi_openllama7b_gowalla.json
