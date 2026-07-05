"""
模型适配器测试示例

配置通过 .env 文件读取，使用前请编辑项目根目录下的 .env 文件。
"""

from tiny_agent import OpenAIAdapter, SystemMessage, HumanMessage
from tiny_agent.config import config


def test_invoke():
    llm = OpenAIAdapter()

    messages = [
        SystemMessage(content="你是一个友好的助手"),
        HumanMessage(content="你好，告诉我一个有趣的事实")
    ]

    response = llm.invoke(messages)
    print("=== 同步调用结果 ===")
    print(f"内容: {response.content}")
    if response.usage:
        print(f"Token 使用: {response.usage.total_tokens}")
    print()


def test_stream():
    llm = OpenAIAdapter()

    messages = [
        SystemMessage(content="你是一个友好的助手"),
        HumanMessage(content="用一句话描述太阳")
    ]

    print("=== 流式调用结果 ===")
    for chunk in llm.stream(messages):
        print(chunk, end="", flush=True)
    print("\n")


def test_tool_call():
    llm = OpenAIAdapter()

    messages = [
        SystemMessage(content="你可以调用工具来回答问题"),
        HumanMessage(content="查询北京的天气")
    ]

    tools = [
        {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    ]

    response = llm.invoke(messages, tools=tools)
    print("=== 工具调用测试 ===")
    if response.has_tool_calls:
        for tc in response.tool_calls:
            print(f"调用工具: {tc.function_name}")
            print(f"参数: {tc.arguments}")
    else:
        print(f"直接回答: {response.content}")


if __name__ == "__main__":
    test_invoke()
    test_stream()
    test_tool_call()
