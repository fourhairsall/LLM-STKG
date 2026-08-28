OUT="D:/databuddy/专利写作/2026年7月/旅游推荐论文/code/_import_probe.out"
: > "$OUT"
PY="C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
run() {
  local LABEL="$1"; shift
  echo "===== $LABEL =====" >> "$OUT"
  env "$@" "$PY" _import_probe.py >> "$OUT" 2>&1
  echo "rc=$? (label=$LABEL)" >> "$OUT"
}
run "A_omp1_mkl1" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
run "B_omp1_mkl1_oblas1" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run "C_numexpr1_oblas1" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
run "D_torch_threads1" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1
echo "PROBE_ALL_DONE" >> "$OUT"
