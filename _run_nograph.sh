#!/bin/bash
cd "D:/databuddy/专利写作/2026年7月/旅游推荐论文/code"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TORCH_NUM_THREADS=1
PY="C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
COMMON="--device cpu --max_degree 10 --batch_size 1024 --lr 4e-3 --epochs 30 \
--bge_model_dir bge_model --bge_cache poi_bge_emb.npy --use_bge --sem_thr 0.90 \
--scorer dot --session_pool mean --use_sgcp --ours_only"
for s in 42 123 777; do
  "$PY" -m llm_stkg.head_to_head $COMMON --no_graph --seed $s \
      --out nograph_s$s.json > nograph_s$s.log 2>&1 &
done
wait
echo "ALL NOGRAPH DONE"
