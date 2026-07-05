"""
工具注册中心

管理所有已注册的工具，支持：
- 装饰器注册：@registry.register
- 转为 OpenAI 工具格式
- 按名称查找并执行
"""

from typing import Dict, List, Callable, Optional, Any

from tiny_agent.core.base import ToolCall, ToolMessage
from tiny_agent.tools.base import Tool


class ToolRegistry:
    """工具注册中心，管理所有可用工具。"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable:
        """
        装饰器：注册一个函数为工具。

        用法：
            registry = ToolRegistry()

            @registry.register(description="获取指定城市的天气")
            def get_weather(city: str) -> str:
                return f"{city}：晴天，25°C"

        Args:
            name: 工具名称，默认使用函数名
            description: 工具描述，默认使用函数 docstring
        """

        def decorator(func: Callable) -> Callable:
            tool = Tool(func, name=name, description=description)
            self._tools[tool.name] = tool
            return func

        return decorator

    def add(self, tool: Tool) -> None:
        """直接添加一个 Tool 实例。"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """按名称获取工具。"""
        return self._tools.get(name)

    def list(self) -> List[Tool]:
        """返回所有已注册的工具列表。"""
        return list(self._tools.values())

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """将所有工具转换为 OpenAI 工具调用格式。"""
        return [tool.to_openai_format() for tool in self._tools.values()]

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """将所有工具转换为字典列表（兼容旧格式）。"""
        return [tool.to_dict() for tool in self._tools.values()]

    def execute(self, tool_call: ToolCall) -> ToolMessage:
        """
        执行一个工具调用并返回 ToolMessage。

        Args:
            tool_call: LLM 返回的工具调用请求

        Returns:
            ToolMessage: 包含执行结果的工具消息

        Raises:
            ValueError: 工具未注册
        """
        tool = self._tools.get(tool_call.function_name)
        if not tool:
            available = ", ".join(self._tools.keys())
            raise ValueError(
                f"Tool '{tool_call.function_name}' not found. Available: {available}"
            )

        result = tool.run(**tool_call.arguments)
        return ToolMessage(content=result, tool_call_id=tool_call.id)

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools.keys())})"
