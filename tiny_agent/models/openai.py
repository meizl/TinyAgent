"""
OpenAI 模型适配器

基于 OpenAI SDK 实现 BaseLLM 接口，支持：
- 兼容 OpenAI API 及所有兼容 OpenAI 协议的模型（DeepSeek、通义千问、智谱等）
- 同步/异步调用
- 流式/非流式输出
- 工具调用（Function Calling）
- 完整的 API 参数支持（temperature、top_p、max_tokens 等）
"""

import os
import json
from typing import List, Optional, Dict, Any, Iterator, AsyncIterator, Literal

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionToolParam

from tiny_agent.core.base import (
    BaseLLM,
    Message,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    ToolCall,
    LLMResponse,
    LLMUsage,
)


class OpenAIAdapter(BaseLLM):
    """OpenAI API 适配器，同时兼容所有基于 OpenAI 协议的第三方模型。"""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        n: int = 1,
        stop: Optional[List[str]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[Dict[str, int]] = None,
        response_format: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
        **kwargs: Any
    ):
        """
        初始化 OpenAI 适配器。

        配置优先级：显式传参 > .env 文件 > 环境变量。
        所有参数均可在调用时通过 invoke/ainvoke/stream/astream 的 **kwargs 覆盖。

        Args:
            model: 模型名称，默认 gpt-3.5-turbo
            api_key: API 密钥，不传则从配置文件/环境变量 OPENAI_API_KEY 读取
            base_url: API 地址，不传则从配置文件/环境变量 OPENAI_BASE_URL 读取（用于第三方兼容模型）
            temperature: 采样温度，0~2，越高越随机
            max_tokens: 最大输出 token 数
            top_p: 核采样，0~1，越小越集中
            n: 生成候选数量
            stop: 停止序列，遇到这些字符串时停止生成
            presence_penalty: 存在惩罚，-2~2，正数减少重复话题
            frequency_penalty: 频率惩罚，-2~2，正数减少重复词
            logit_bias: 对数偏差，控制特定 token 的生成概率
            response_format: 响应格式，如 {"type": "json_object"}
            seed: 随机种子，用于复现结果
            **kwargs: 传递给 OpenAI 客户端的其他参数
        """
        # 尝试从 config 读取默认值
        _api_key = api_key
        _base_url = base_url
        _model = model

        try:
            from tiny_agent.config import config
            _api_key = _api_key or config.api_key
            _base_url = _base_url if _base_url is not None else config.base_url
            if model == "gpt-3.5-turbo":
                _model = config.default_model
        except ImportError:
            pass

        # 兜底：从环境变量读取
        self.api_key = _api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = _base_url or os.getenv("OPENAI_BASE_URL") or None
        self.model = _model

        # 默认参数
        self.default_params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "n": n,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "response_format": response_format,
            "seed": seed,
        }

        if not self.api_key:
            raise ValueError(
                "api_key must be provided or set via OPENAI_API_KEY environment variable"
            )

        # 同步客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, **kwargs)
        # 异步客户端
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, **kwargs)

    def _get_call_params(self, **kwargs) -> Dict[str, Any]:
        """合并默认参数与调用时传入的参数，调用时参数优先。"""
        params = {k: v for k, v in self.default_params.items() if v is not None}
        params.update(kwargs)
        return params

    def set_model(self, model: str) -> None:
        """动态切换模型，不需要重建客户端连接。"""
        self.model = model

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """将内部 Message 对象列表转换为 OpenAI API 所需的消息格式。"""
        converted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    tool_calls = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function_name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                    converted.append({
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": tool_calls,
                    })
                else:
                    converted.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                converted.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                })
            else:
                converted.append({"role": "user", "content": str(msg.content)})
        return converted

    def _convert_tools(
        self, tools: Optional[List[Dict]]
    ) -> Optional[List[ChatCompletionToolParam]]:
        """将工具描述字典列表转换为 OpenAI API 的工具参数格式。"""
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            }
            for tool in tools
        ]

    def _parse_response(self, response) -> LLMResponse:
        """将 OpenAI API 原始响应解析为统一的 LLMResponse 对象。"""
        content = response.choices[0].message.content or ""
        tool_calls = []

        # 解析工具调用
        if response.choices[0].message.tool_calls:
            for tc in response.choices[0].message.tool_calls:
                arguments = {}
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    arguments = tc.function.arguments
                tool_calls.append(
                    ToolCall(id=tc.id, function_name=tc.function.name, arguments=arguments)
                )

        # 解析 token 用量
        usage = None
        if response.usage:
            usage = LLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return LLMResponse(content=content, tool_calls=tool_calls, usage=usage)

    # ── 同步调用 ──────────────────────────────────────────────

    def invoke(
        self, messages: List[Message], tools: Optional[List[Dict]] = None, **kwargs
    ) -> LLMResponse:
        """
        同步调用 LLM，返回完整响应。

        Args:
            messages: 消息列表
            tools: 可用工具列表
            **kwargs: 调用时覆盖的参数（temperature、max_tokens、top_p 等）

        Returns:
            LLMResponse: 统一的响应对象
        """
        converted_messages = self._convert_messages(messages)
        converted_tools = self._convert_tools(tools)
        params = self._get_call_params(**kwargs)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=converted_messages,
            tools=converted_tools,
            **params,
        )

        return self._parse_response(response)

    # ── 异步调用 ──────────────────────────────────────────────

    async def ainvoke(
        self, messages: List[Message], tools: Optional[List[Dict]] = None, **kwargs
    ) -> LLMResponse:
        """
        异步调用 LLM，返回完整响应。

        Args:
            messages: 消息列表
            tools: 可用工具列表
            **kwargs: 调用时覆盖的参数（temperature、max_tokens、top_p 等）

        Returns:
            LLMResponse: 统一的响应对象
        """
        converted_messages = self._convert_messages(messages)
        converted_tools = self._convert_tools(tools)
        params = self._get_call_params(**kwargs)

        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=converted_messages,
            tools=converted_tools,
            **params,
        )

        return self._parse_response(response)

    # ── 同步流式 ──────────────────────────────────────────────

    def stream(
        self, messages: List[Message], tools: Optional[List[Dict]] = None, **kwargs
    ) -> Iterator[str]:
        """
        同步流式调用 LLM，逐 token 返回文本内容。

        Args:
            messages: 消息列表
            tools: 可用工具列表
            **kwargs: 调用时覆盖的参数（temperature、max_tokens、top_p 等）

        Yields:
            str: 逐 token 的文本内容
        """
        converted_messages = self._convert_messages(messages)
        converted_tools = self._convert_tools(tools)
        params = self._get_call_params(**kwargs)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=converted_messages,
            tools=converted_tools,
            stream=True,
            **params,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ── 异步流式 ──────────────────────────────────────────────

    async def astream(
        self, messages: List[Message], tools: Optional[List[Dict]] = None, **kwargs
    ) -> AsyncIterator[str]:
        """
        异步流式调用 LLM，逐 token 返回文本内容。

        Args:
            messages: 消息列表
            tools: 可用工具列表
            **kwargs: 调用时覆盖的参数（temperature、max_tokens、top_p 等）

        Yields:
            str: 逐 token 的文本内容
        """
        converted_messages = self._convert_messages(messages)
        converted_tools = self._convert_tools(tools)
        params = self._get_call_params(**kwargs)

        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=converted_messages,
            tools=converted_tools,
            stream=True,
            **params,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
