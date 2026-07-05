"""Document 数据类"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Document:
    """RAG 文档对象"""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        source = self.metadata.get("source", "unknown")
        preview = self.text[:50].replace("\n", " ")
        return f"Document(source={source}, text='{preview}...')"
