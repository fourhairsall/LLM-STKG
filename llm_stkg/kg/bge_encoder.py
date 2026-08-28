"""BGE 本地语义编码器（C1 真实 LLM 嵌入落地）。

用本地 sentence-transformers 加载已下载的 BAAI/bge-base-en-v1.5（code/bge_model 目录），
将 POI 文本编码为 768 维语义向量，替换原 MD5 哈希占位（32 维噪声），使语义边与语义表征
具备真实世界知识。延迟加载：首次 encode 时才载入模型，避免 import 阶段占用显存/内存。
"""
import numpy as np


class BGESemanticEncoder:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self._model = None
        self.dim = 768

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_dir)
            self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size=256):
        self._ensure()
        vecs = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float64)
