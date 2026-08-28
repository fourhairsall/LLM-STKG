OUT="D:/databuddy/专利写作/2026年7月/旅游推荐论文/code/_install_bge2.out"
: > "$OUT"
PY="C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="D:/databuddy/专利写作/2026年7月/旅游推荐论文/code/.hf_cache"
rm -rf "$HF_HOME/models--BAAI--bge-base-en-v1.5"
run_one() {
  "$PY" -c "
from huggingface_hub import snapshot_download
p = snapshot_download('BAAI/bge-base-en-v1.5')
print('SNAP', p)
from sentence_transformers import SentenceTransformer
import numpy as np
m = SentenceTransformer(p)
v = m.encode(['central park new york','empire state building'], normalize_embeddings=True)
print('DIM', m.get_sentence_embedding_dimension(), 'SIM', float(np.dot(v[0], v[1])))
" > _bge_run.log 2>&1
  return $?
}
for i in $(seq 1 8); do
  if run_one; then cat _bge_run.log >> "$OUT"; echo "BGE_OK attempt=$i"; break; fi
  echo "retry $i rc=$?" >> "$OUT"; tail -3 _bge_run.log >> "$OUT"; sleep 3
done
echo "INSTALL_BGE2_DONE" >> "$OUT"
