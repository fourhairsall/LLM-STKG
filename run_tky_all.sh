#!/usr/bin/env bash
# Foursquare-TKY 全流程：数据准备 → BGE 语义缓存 → 头对头评估（ours 全配置 + 基线）。
# 用法：bash run_tky_all.sh
# 前置：dataset_TSMC2014_TKY.csv 已下载到 data/real_foursquare_tky/
set -e
cd "$(dirname "$0")"   # 切到 code 目录

# 线程前缀（防 torch 在沙箱内 segfault，缺一不可）
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TORCH_NUM_THREADS=1

PY="C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
PROC="../../data/real_foursquare_tky/processed"

echo "=== [1/3] TKY 数据准备 ==="
"$PY" real_data_prepare_tky.py

echo "=== [2/3] TKY BGE 语义缓存 ==="
if [ -f poi_bge_emb_tky.npy ]; then
  echo "BGE 缓存已存在，跳过"
else
  "$PY" prepare_tky_bge.py
fi

echo "=== [3/3] TKY 头对头评估（ours 全配置 + 基线）==="
"$PY" -m llm_stkg.head_to_head \
  --dataset foursquare \
  --processed_dir "$PROC" \
  --device cuda \
  --use_bge --sem_thr 0.90 --bge_model_dir bge_model --bge_cache poi_bge_emb_tky.npy \
  --scorer dot --session_pool mean --use_sgcp \
  --hist_mode user --seq_len 200 \
  --prior_channels cnt,rec,pop --gate_mode context \
  --batch_size 1024 --lr 4e-3 --epochs 30 \
  --save_model tky_c6u_seed42.pt --out tky_head_to_head.json

echo "=== TKY 全流程完成 ==="
