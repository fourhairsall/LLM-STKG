OUT="D:/databuddy/专利写作/2026年7月/旅游推荐论文/code/_test_envs.out"
: > "$OUT"
ENVS="default sph patent_env patent_env2 litecls na_verify"
for ENV in $ENVS; do
  PY="C:/Users/Lenovo/.workbuddy/binaries/python/envs/$ENV/Scripts/python.exe"
  if [ ! -f "$PY" ]; then echo "=== $ENV: NO PYTHON ===" >> "$OUT"; continue; fi
  for OMP in "" "1"; do
    LABEL="$ENV omp=${OMP:-default}"
    echo "=== $LABEL ===" >> "$OUT"
    if [ -z "$OMP" ]; then
      "$PY" -c "import torch;print('torch',torch.__version__)" >> "$OUT" 2>&1 || echo "rc=$?" >> "$OUT"
    else
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -c "import torch;print('torch',torch.__version__)" >> "$OUT" 2>&1 || echo "rc=$?" >> "$OUT"
    fi
  done
done
echo "ALL_DONE" >> "$OUT"
