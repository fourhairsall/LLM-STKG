OUT="D:/databuddy/专利写作/2026年7月/旅游推荐论文/code/_dl_bge_weights.out"
: > "$OUT"
URL="https://hf-mirror.com/BAAI/bge-base-en-v1.5/resolve/main/model.safetensors"
DST="D:/databuddy/专利写作/2026年7月/旅游推荐论文/code/bge_model/model.safetensors"
EXP=437955512
for a in $(seq 1 10); do
  curl -sL --max-time 1800 -w "HTTP=%{http_code} SIZE=%{size_download}\n" -o "$DST" "$URL" >> "$OUT" 2>&1
  sz=$(stat -c%s "$DST" 2>/dev/null || echo 0)
  echo "attempt=$a got=$sz expected=$EXP" >> "$OUT"
  if [ "$sz" -eq "$EXP" ]; then echo "WEIGHTS_OK attempt=$a" >> "$OUT"; break; fi
  echo "retry $a (size mismatch)" >> "$OUT"; sleep 2
done
echo "DL_WEIGHTS_DONE" >> "$OUT"
