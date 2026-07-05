"""
工具基类

参考 LangChain 的 Tool 设计，每个工具包含：
- 函数体
- 名称、描述
- 参数 JSON Schema（自动从函数签名推断）
"""

import inspect
import json
from typing import Callable, Dict, Any, Optional, get_type_hints

from tiny_agent.core.base import ToolCall, ToolMessage

# Python 类型 → JSON Schema 类型映射
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _infer_schema(func: Callable) -> Dict[str, Any]:
    """从函数签名自动推断 JSON Schema。"""
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        python_type = hints.get(name, str)
        json_type = _TYPE_MAP.get(python_type, "string")

        prop = {"type": json_type}
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(name)

        properties[name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


class Tool:
    """工具封装：一个可调用的函数 + 名称 + 描述 + 参数 Schema。"""

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            func: 工具函数
            name: 工具名称，默认使用函数名
            description: 工具描述，默认使用函数 docstring 第一行
            parameters: 参数 JSON Schema，默认从函数签名自动推断
        """
        self.func = func
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "").strip().split("\n")[0]
        self.parameters = parameters or _infer_schema(func)

    def run(self, **kwargs) -> str:
        """执行工具并返回字符串结果。"""
        result = self.func(**kwargs)
        return str(result)

    def to_openai_format(self) -> Dict[str, Any]:
        """转换为 OpenAI 工具调用格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（兼容旧的工具格式）。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def __repr__(self) -> str:
        return f"Tool(name={self.name})"
