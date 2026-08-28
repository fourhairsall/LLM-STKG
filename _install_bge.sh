OUT="D:/databuddy/专利写作/2026年7月/旅游推荐论文/code/_install_bge.out"
: > "$OUT"
PY="C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
PIP="C:/Users/Lenovo/.workbuddy/binaries/python/envs/default/Scripts/pip.exe"
echo "=== pip install sentence-transformers ===" >> "$OUT"
"$PIP" install -q sentence-transformers 2>&1 | tail -5 >> "$OUT"
echo "rc_pip=$?" >> "$OUT"
echo "=== download bge-base-en-v1.5 via hf-mirror ===" >> "$OUT"
HF_ENDPOINT=https://hf-mirror.com "$PY" -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('BAAI/bge-base-en-v1.5')
print('BGELoaded dim=', m.get_sentence_embedding_dimension())
import numpy as np
v = m.encode(['central park new york','empire state building'], normalize_embeddings=True)
print('enc shape', v.shape, 'sim', float(np.dot(v[0],v[1])))
" 2>&1 | tail -15 >> "$OUT"
echo "INSTALL_BGE_DONE" >> "$OUT"
