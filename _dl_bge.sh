OUT="D:/databuddy/专利写作/2026年7月/旅游推荐论文/code/_dl_bge.out"
: > "$OUT"
BASE="https://hf-mirror.com/BAAI/bge-base-en-v1.5/resolve/main"
DST="D:/databuddy/专利写作/2026年7月/旅游推荐论文/code/bge_model"
mkdir -p "$DST/1_Pooling"
FILES="config.json modules.json config_sentence_transformers.json sentence_bert_config.json 1_Pooling/config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt README.md .gitattributes"
dl() {
  local rel="$1"; local full="$DST/$rel"
  mkdir -p "$(dirname "$full")"
  for a in $(seq 1 6); do
    curl -sL --max-time 120 -w "HTTP=%{http_code} SIZE=%{size_download}\n" -o "$full" "$BASE/$rel" >> "$OUT" 2>&1
    local sz=$(stat -c%s "$full" 2>/dev/null || echo 0)
    if [ "$sz" -gt 0 ]; then echo "OK $rel ($sz bytes) attempt=$a" >> "$OUT"; return 0; fi
    echo "retry $rel attempt=$a sz=$sz" >> "$OUT"; sleep 3
  done
  echo "FAIL $rel" >> "$OUT"; return 1
}
for f in $FILES; do dl "$f"; done
echo "DL_BGE_DONE" >> "$OUT"
