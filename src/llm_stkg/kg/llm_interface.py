"""LLM 接口：真实部署时替换为 GPT/GLM API 调用；本地回退用确定性文本特征，保证可离线复现。

创新点呼应（论文 C1 / C3）：
- extract_relations(): 真实场景用「因式分解提示」让 LLM 抽取 POI 间互补/相似/层级关系，补全旅游知识图谱；
  本地回退用文本重叠相似度，便于无 GPU/无 API 时跑通整条流水线。
- text_embedding(): 为 POI 描述/评论生成语义向量，作为语义边与意图推理的输入。
"""
import hashlib
import re
from collections import Counter


class LLMInterface:
    def __init__(self, use_api: bool = False, api_key: str = None, bge_model_dir: str = None):
        self.use_api = use_api
        self.api_key = api_key
        self.bge_model_dir = bge_model_dir
        self._bge = None
        self._vocab = None

    # ---------- 真实 API 接入点（占位，按需实现） ----------
    def _call_llm(self, prompt: str) -> str:
        # TODO: 接入 OpenAI / 智谱 GLM / 通义千问等；
        # 例如 returns client.chat.completions.create(...).choices[0].message.content
        raise NotImplementedError("设置 use_api=True 并实现 _call_llm 以接入真实大模型")

    def llm_relation_prompt(self, poi_a_text: str, poi_b_text: str) -> str:
        return (
            f"给定两个旅游 POI 的描述：\nA: {poi_a_text}\nB: {poi_b_text}\n"
            "请判断二者关系：互补/相似/层级/无关，并给 0~1 相似度。仅输出 JSON。"
        )

    # ---------- 本地确定性回退（可复现） ----------
    def _tokenize(self, text):
        return re.findall(r"[a-z0-9\u4e00-\u9fff]+", (text or "").lower())

    def text_embedding(self, texts, dim=None):
        # C1 真实 LLM 嵌入路径：用本地 BGE 编码，取代 MD5 哈希占位（32 维噪声）。
        # 返回维度由模型决定（bge-base=768），与下游动态 feat_dim 兼容。
        if self.bge_model_dir is not None:
            from .bge_encoder import BGESemanticEncoder
            if self._bge is None:
                self._bge = BGESemanticEncoder(self.bge_model_dir)
            return self._bge.encode(texts).tolist()
        if dim is None:
            dim = 32
        vecs = []
        for t in texts:
            toks = self._tokenize(t)
            v = [0.0] * dim
            for w in toks:
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % dim] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            v = [x / norm for x in v]
            vecs.append(v)
        return vecs

    def extract_relations(self, poi_a_text, poi_b_text):
        if self.use_api:
            return self._api_relation(poi_a_text, poi_b_text)
        ta = set(self._tokenize(poi_a_text))
        tb = set(self._tokenize(poi_b_text))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _api_relation(self, a, b):
        out = self._call_llm(self.llm_relation_prompt(a, b))
        # TODO: 解析 JSON 取相似度
        return 0.0
