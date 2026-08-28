#!/usr/bin/env bash
# 跨域新数据集：LLM4POI-style v4 —— 剩余 3 配置（steam200k_nobge / amazon_beauty text+nobge）
# 提速：--no_grad_ckpt（微基准 0.49 -> 0.28 s/step）；受后台跟踪；带时间戳日志。
# 协议不变：epochs=1 aug_max=2 batch=4 grad_accum=4 lr=1e-4（与已完成的 steam200k text 一致，可比）。
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TORCH_NUM_THREADS=1
cd "/d/databuddy/专利写作/2026年7月/旅游推荐论文/code"
PY="C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
DATA_ROOT="D:/databuddy/专利写作/2026年7月/旅游推荐论文/data"
LOG="_llm4poi_v4.log"
: > "$LOG"

run() {
  local cfg=$1; local city=$2; shift 2
  local log="_llm4poi_v4_${cfg}.log"
  echo "[$(date +%H:%M:%S)] START $cfg (city=$city)" >> "$LOG"
  "$PY" -u llm4poi_baseline.py --city "$city" \
      --data_root "$DATA_ROOT/$city/processed" \
      --epochs 1 --aug_max 2 --batch 4 --grad_accum 4 --lr 1e-4 \
      --no_grad_ckpt "$@" --out "llm4poi_${cfg}.json" \
      > "$log" 2>&1
  local ec=$?
  local ok=NO; [ -f "llm4poi_${cfg}.json" ] && ok=YES
  echo "[$(date +%H:%M:%S)] END $cfg exit=$ec json=$ok" >> "$LOG"
}

run steam200k_nobge steam200k --no_bge
run amazonbeauty amazon_beauty
run amazonbeauty_nobge amazon_beauty --no_bge
echo "[$(date +%H:%M:%S)] ALL_LLM4POI_V4_DONE" >> "$LOG"
