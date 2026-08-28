#!/usr/bin/env bash
# 跨域新增数据集：head_to_head（ours + 6 基线）训练评测
# 单 GPU 顺序执行，规避显存争用。带线程前缀防 segfault。
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TORCH_NUM_THREADS=1
cd "/d/databuddy/专利写作/2026年7月/旅游推荐论文/code"
PY="C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe"

echo "===== [1/2] head_to_head steam200k (epochs=20) ====="
"$PY" -u -m llm_stkg.head_to_head --dataset steam200k --device cuda --epochs 20 \
    --out head_to_head_steam200k.json --ds_max_pois 5000 --ds_max_users 20000 \
    > _cross_steam200k.log 2>&1
echo "EXIT_steam200k=$?"

echo "===== [2/2] head_to_head amazon_beauty (epochs=20) ====="
"$PY" -u -m llm_stkg.head_to_head --dataset amazon_beauty --device cuda --epochs 20 \
    --out head_to_head_amazonbeauty.json --ds_max_pois 5000 --ds_max_users 20000 \
    > _cross_amazonbeauty.log 2>&1
echo "EXIT_amazonbeauty=$?"

echo "ALL_HEADTOHEAD_DONE"
