#!/usr/bin/env bash
# 跨域新数据集：LLM4POI-style 生成式基线（文本种子 + ID-only 因果对照） v2
# 与 v1 的差异（提速，公平统一）：
#   * epochs 3 -> 2（因果对照只需显示差异方向，无需 3 epoch 高精度）
#   * aug_max 8 -> 4（训练样本减半；有效 batch 仍 = batch*grad_accum = 16，不冒 OOM）
#   * 故单配置约 4x 加速，4 个配置串行总时长可控
# 协议：mask_history 由 llm4poi_baseline.py 按测试端重访率自动判定（与 head_to_head 一致）。
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TORCH_NUM_THREADS=1
cd "/d/databuddy/专利写作/2026年7月/旅游推荐论文/code"
PY="C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe"

# 保险：清掉任何残留的 llm4poi python（孤儿进程）
taskkill //F //PID 8492 >/dev/null 2>&1 || true
pkill -f "llm4poi_baseline" >/dev/null 2>&1 || true

DATA_ROOT="D:/databuddy/专利写作/2026年7月/旅游推荐论文/data"
for city in steam200k amazon_beauty; do
  for mode in text nobge; do
    if [ "$mode" = "text" ]; then
      extra=""
      out="llm4poi_${city}.json"
    else
      extra="--no_bge"
      out="llm4poi_${city}_nobge.json"
    fi
    echo "===== [LLM4POI-style v2] $city ($mode) ====="
    "$PY" -u llm4poi_baseline.py --city "$city" \
        --data_root "$DATA_ROOT/$city/processed" \
        --epochs 2 --aug_max 4 --batch 4 --grad_accum 4 --lr 1e-4 \
        $extra --out "$out" \
        > "_llm4poi_v2_${city}_${mode}.log" 2>&1
    echo "EXIT_${city}_${mode}=$?"
  done
done
echo "ALL_LLM4POI_V2_DONE"
