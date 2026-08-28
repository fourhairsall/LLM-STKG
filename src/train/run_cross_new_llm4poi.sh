#!/usr/bin/env bash
# 跨域新数据集：LLM4POI-style 生成式基线（文本种子 + ID-only 因果对照）
# 设计：
#   * 文本可得域（steam200k / amazon_beauty）应使 LLM4POI-style（BGE 语义种子）非零；
#   * 同域 ID-only（--no_bge，无语义种子）应≈0 —— 镜像 ML-1M 的因果证据
#     （0.1865 text vs 0.0000 ID-only），证明"文本可得 → LLM4POI-style 生效"是因果的；
#   * 与 head_to_head 共享同一 GPU，故先等待 head_to_head 进程退出再开始，避免显存争用。
# 协议：mask_history 由脚本按测试端重访率自动判定（与 head_to_head 完全一致）。
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TORCH_NUM_THREADS=1
cd "/d/databuddy/专利写作/2026年7月/旅游推荐论文/code"
PY="C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe"

echo "[wait] 等待 head_to_head 后台进程释放 GPU ..."
while pgrep -f "llm_stkg.head_to_head" >/dev/null 2>&1; do sleep 30; done
echo "[wait] GPU 已空闲，开始 LLM4POI-style 基线"

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
    echo "===== [LLM4POI-style] $city ($mode) ====="
    "$PY" -u llm4poi_baseline.py --city "$city" \
        --data_root "$DATA_ROOT/$city/processed" \
        --epochs 3 --aug_max 8 --batch 4 --grad_accum 4 --lr 1e-4 \
        $extra --out "$out" \
        > "_llm4poi_${city}_${mode}.log" 2>&1
    echo "EXIT_${city}_${mode}=$?"
  done
done
echo "ALL_LLM4POI_DONE"
