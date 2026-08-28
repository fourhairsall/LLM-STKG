#!/usr/bin/env bash
# 跨域新数据集：LLM4POI-style 生成式基线 v3（受跟踪、自动等 GPU、提速）
# 提速 vs v2：aug_max 4->2（训练样本减半），epochs 2->1（基线只需显示方向）。
# 协议：mask_history 由 llm4poi_baseline.py 按测试端重访率自动判定（与 head_to_head 一致）。
# 先等待 amazon_beauty head_to_head（6g90ps）释放 GPU，再顺序跑 4 配置。
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TORCH_NUM_THREADS=1
cd "/d/databuddy/专利写作/2026年7月/旅游推荐论文/code"
PY="C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
DATA_ROOT="D:/databuddy/专利写作/2026年7月/旅游推荐论文/data"

echo "[v3] waiting for head_to_head GPU to free ..."
while pgrep -f "llm_stkg.head_to_head" >/dev/null 2>&1; do sleep 20; done
echo "[v3] GPU free, starting LLM4POI 4-config (epochs=1 aug_max=2)"

for city in steam200k amazon_beauty; do
  for mode in text nobge; do
    if [ "$mode" = "text" ]; then
      extra=""
      out="llm4poi_${city}.json"
    else
      extra="--no_bge"
      out="llm4poi_${city}_nobge.json"
    fi
    echo "===== [LLM4POI-style v3] $city ($mode) ====="
    "$PY" -u llm4poi_baseline.py --city "$city" \
        --data_root "$DATA_ROOT/$city/processed" \
        --epochs 1 --aug_max 2 --batch 4 --grad_accum 4 --lr 1e-4 \
        $extra --out "$out" \
        > "_llm4poi_v3_${city}_${mode}.log" 2>&1
    echo "EXIT_${city}_${mode}=$?"
  done
done
echo "ALL_LLM4POI_V3_DONE"
