"""文本切分器：多种切分策略"""

from typing import List
from tiny_agent.rag.document import Document


class BaseTextSplitter:
    """切分器基类"""

    def split_text(self, text: str) -> List[str]:
        raise NotImplementedError

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将一批文档切分为更小的文档块"""
        result = []
        for doc in documents:
            chunks = self.split_text(doc.text)
            for i, chunk in enumerate(chunks):
                metadata = {**doc.metadata, "chunk_index": i, "chunk_count": len(chunks)}
                result.append(Document(text=chunk, metadata=metadata))
        return result


class CharacterTextSplitter(BaseTextSplitter):
    """按字符数固定切分"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks


class RecursiveCharacterTextSplitter(BaseTextSplitter):
    """递归字符切分器：优先按段落、换行、句子、字符逐级切分，避免切断句子"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._separators = ["\n\n", "\n", "。", ".", "！", "？", "；", ";", " "]

    def split_text(self, text: str) -> List[str]:
        return self._recursive_split(text, self._separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """递归切分：优先用高级分隔符，切不动的再降级"""
        chunks = []
        # 找到一个能切开的分隔符
        separator = self._separators[-1]  # 兜底：单字符切分
        for sep in separators:
            if sep in text:
                separator = sep
                break

        # 按分隔符切分，再合并不超过 chunk_size 的片段
        parts = text.split(separator)
        current = ""
        for part in parts:
            if len(current) + len(part) + len(separator) <= self.chunk_size:
                current = (current + separator + part).lstrip(separator) if current else part
            else:
                if current:
                    chunks.append(current)
                # 单个片段超过 chunk_size，降级切分
                if len(part) > self.chunk_size:
                    chunks.extend(self._recursive_split(part, separators[1:]))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks


class ParagraphTextSplitter(BaseTextSplitter):
    """段落切分器：按双换行切分，保持段落完整性"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) <= self.chunk_size:
                current = (current + "\n\n" + para).strip() if current else para
            else:
                if current:
                    chunks.append(current)
                current = para
        if current:
            chunks.append(current)
        return chunks
