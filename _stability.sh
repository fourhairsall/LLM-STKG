OUT="D:/databuddy/专利写作/2026年7月/旅游推荐论文/code/_stability.out"
: > "$OUT"
PY="C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
ok=0; fail=0
for i in $(seq 1 10); do
  out=$(OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TORCH_NUM_THREADS=1 "$PY" -c "import torch; a=torch.randn(8,8); b=a@a; print('OK%d'%$i)" 2>&1)
  if echo "$out" | grep -q "OK"; then ok=$((ok+1)); else fail=$((fail+1)); echo "RUN$i FAIL: $out" >> "$OUT"; fi
done
echo "STABLE ok=$ok fail=$fail" >> "$OUT"
