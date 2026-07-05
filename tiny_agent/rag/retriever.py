"""
Retriever — RAG 统一入口

对外暴露的核心类，封装了加载 → 切分 → 向量化 → 存储 → 检索 的完整流程。

使用示例：
    from tiny_agent.rag import Retriever

    retriever = Retriever()
    retriever.add_file("doc.pdf")
    retriever.add_texts(["知识内容1", "知识内容2"])

    docs = retriever.search("查询内容", top_k=3)
"""

from typing import List, Optional
from tiny_agent.rag.document import Document
from tiny_agent.rag.loaders import load_file, load_directory
from tiny_agent.rag.splitters import BaseTextSplitter, RecursiveCharacterTextSplitter
from tiny_agent.rag.embeddings import BaseEmbeddings, OpenAIEmbeddings
from tiny_agent.rag.stores import BaseStore, InMemoryStore


class Retriever:
    """RAG 检索器：一站式文档索引与搜索"""

    def __init__(
        self,
        embeddings: Optional[BaseEmbeddings] = None,
        store: Optional[BaseStore] = None,
        splitter: Optional[BaseTextSplitter] = None,
    ):
        """
        Args:
            embeddings: Embedding 模型，默认 OpenAIEmbeddings
            store: 向量存储，默认 InMemoryStore
            splitter: 文本切分器，默认 RecursiveCharacterTextSplitter
        """
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.store = store or InMemoryStore()
        self.splitter = splitter or RecursiveCharacterTextSplitter()

    # ── 索引（离线阶段） ───────────────────────────────────

    def add_file(self, file_path: str) -> None:
        """索引一个文件（自动识别 PDF/Word/TXT）"""
        documents = load_file(file_path)
        self._index_documents(documents)

    def add_directory(self, dir_path: str, recursive: bool = False) -> None:
        """索引目录下所有支持的文档"""
        documents = load_directory(dir_path, recursive=recursive)
        self._index_documents(documents)

    def add_texts(self, texts: List[str], metadata: Optional[dict] = None) -> None:
        """直接索引文本列表"""
        documents = [Document(text=t, metadata=metadata or {}) for t in texts]
        self._index_documents(documents)

    def _index_documents(self, documents: List[Document]) -> None:
        """内部：切分 → 向量化 → 入库"""
        chunks = self.splitter.split_documents(documents)
        texts = [doc.text for doc in chunks]
        vectors = self.embeddings.embed_batch(texts)
        self.store.add(chunks, vectors)

    # ── 检索（在线阶段） ───────────────────────────────────

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Document]:
        """
        检索最相关的文档块。

        Args:
            query: 查询文本
            top_k: 返回前 k 个结果
            min_score: 最低相似度阈值（0~1）

        Returns:
            相关文档列表（按相似度降序）
        """
        query_vector = self.embeddings.embed(query)
        results = self.store.search(query_vector, top_k=top_k)

        documents = []
        for doc, score in results:
            if score >= min_score:
                doc.metadata["score"] = round(score, 4)
                documents.append(doc)

        return documents

    def search_with_scores(self, query: str, top_k: int = 5) -> List[tuple]:
        """检索并返回带分数的结果"""
        query_vector = self.embeddings.embed(query)
        return self.store.search(query_vector, top_k=top_k)

    # ── 工具方法 ───────────────────────────────────────────

    def clear(self) -> None:
        """清空所有索引"""
        self.store.clear()

    def __len__(self) -> int:
        return len(self.store)

    def __repr__(self) -> str:
        return f"Retriever(embeddings={self.embeddings.__class__.__name__}, store={self.store.__class__.__name__}, chunks={len(self)})"
