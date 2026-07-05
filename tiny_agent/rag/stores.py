"""
向量存储

支持随意切换存储后端：
- InMemoryStore：内存存储（默认，零依赖）
- 可扩展：Chroma、FAISS、Milvus 等
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from tiny_agent.rag.document import Document


class BaseStore(ABC):
    """向量存储基类"""

    @abstractmethod
    def add(self, documents: List[Document], vectors: List[List[float]]) -> None:
        """添加文档和对应的向量"""
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[Document, float]]:
        """检索最相似的 top_k 个文档，返回 (文档, 相似度分数)"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空存储"""
        pass


class InMemoryStore(BaseStore):
    """内存向量存储：使用余弦相似度"""

    def __init__(self):
        self._documents: List[Document] = []
        self._vectors: List[List[float]] = []

    def add(self, documents: List[Document], vectors: List[List[float]]) -> None:
        self._documents.extend(documents)
        self._vectors.extend(vectors)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[Document, float]]:
        if not self._vectors:
            return []

        scores = []
        for i, vec in enumerate(self._vectors):
            score = self._cosine_similarity(query_vector, vec)
            scores.append((self._documents[i], score, i))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score, _ in scores[:top_k]]

    def clear(self) -> None:
        self._documents.clear()
        self._vectors.clear()

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def __len__(self) -> int:
        return len(self._documents)
