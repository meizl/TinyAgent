from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator, AsyncIterator, Union
import json


@dataclass
class ToolCall:
    id: str
    function_name: str
    arguments: Dict[str, Any]


@dataclass
class Message:
    content: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content}


@dataclass
class SystemMessage(Message):
    role: str = "system"
    
    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class HumanMessage(Message):
    role: str = "user"
    
    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class AIMessage(Message):
    role: str = "assistant"
    tool_calls: List[ToolCall] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function_name,
                        "arguments": json.dumps(tc.arguments)
                    }
                }
                for tc in self.tool_calls
            ]
        return result


@dataclass
class ToolMessage(Message):
    role: str = "tool"
    tool_call_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_call_id": self.tool_call_id
        }


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Optional[LLMUsage] = None
    
    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class BaseLLM(ABC):
    
    @abstractmethod
    def invoke(self, messages: List[Message], tools: Optional[List[Dict]] = None) -> LLMResponse:
        pass
    
    @abstractmethod
    async def ainvoke(self, messages: List[Message], tools: Optional[List[Dict]] = None) -> LLMResponse:
        pass
    
    @abstractmethod
    def stream(self, messages: List[Message], tools: Optional[List[Dict]] = None) -> Iterator[str]:
        pass
    
    @abstractmethod
    async def astream(self, messages: List[Message], tools: Optional[List[Dict]] = None) -> AsyncIterator[str]:
        pass
