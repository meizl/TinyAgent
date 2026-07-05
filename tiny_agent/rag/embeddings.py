"""
Embedding 适配器

支持：
- OpenAIEmbeddings：使用 OpenAI 兼容 API
- 自动复用 OpenAIAdapter 的 api_key / base_url 配置
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddings(ABC):
    """Embedding 模型基类"""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """将单条文本转为向量"""
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量向量化（默认逐条调用，子类可重写为真正的批量）"""
        return [self.embed(t) for t in texts]


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI 兼容 Embedding 模型"""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = None,
        base_url: str = None,
    ):
        """
        Args:
            model: Embedding 模型名
            api_key: API Key，默认复用 LLM 配置
            base_url: API 地址，默认复用 LLM 配置
                     注意：DeepSeek 等部分厂商不支持 Embedding，
                     需要单独指向 OpenAI 或其他兼容端点
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

        try:
            from tiny_agent.config import config
            self.api_key = self.api_key or config.api_key
            # Embedding 用独立 base_url，不混用 LLM 的
            self.base_url = self.base_url or config.base_url
        except ImportError:
            pass

        import os
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = self.base_url or os.getenv("OPENAI_BASE_URL") or None

        if not self.api_key:
            raise ValueError("api_key 未配置")

        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def embed(self, text: str) -> List[float]:
        result = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return result.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """真正的批量调用"""
        result = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [d.embedding for d in result.data]
