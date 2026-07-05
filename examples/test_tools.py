"""
工具注册与调用测试

演示：
1. 通过装饰器注册工具
2. LLM 调用工具并执行
3. 完整的 Agent 工具调用循环
"""

from tiny_agent import OpenAIAdapter, SystemMessage, HumanMessage, AIMessage
from tiny_agent.tools import ToolRegistry


# ── 创建注册中心并注册工具 ──────────────────────────────

registry = ToolRegistry()


@registry.register(description="获取指定城市的天气")
def get_weather(city: str) -> str:
    """查询城市天气"""
    weather_data = {
        "北京": "晴天，25°C，湿度 40%",
        "上海": "多云，28°C，湿度 65%",
        "深圳": "阵雨，30°C，湿度 80%",
    }
    return weather_data.get(city, f"{city}：暂无天气数据")


@registry.register(description="计算数学表达式")
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


def test_tool_registry():
    """测试工具注册"""
    print("=== 工具注册中心 ===")
    print(f"已注册工具: {registry}")
    print(f"工具数量: {len(registry)}")
    print()

    for tool in registry.list():
        print(f"  - {tool.name}: {tool.description}")


def test_tool_call_with_llm():
    """测试 LLM 调用工具"""
    llm = OpenAIAdapter()

    messages = [
        SystemMessage(content="你可以调用工具来回答用户问题"),
        HumanMessage(content="查询北京的天气"),
    ]

    # 传入 ToolRegistry（自动转为 OpenAI 格式）
    response = llm.invoke(messages, tools=registry)
    print("\n=== LLM 工具调用 ===")

    if response.has_tool_calls:
        for tc in response.tool_calls:
            print(f"LLM 调用: {tc.function_name}({tc.arguments})")
            
            # 执行工具
            result = registry.execute(tc)
            print(f"工具返回: {result.content}")
    else:
        print(f"直接回答: {response.content}")


def test_agent_loop():
    """完整的 Agent 循环：思考 → 工具调用 → 结果注入 → 继续思考"""
    llm = OpenAIAdapter()
    
    messages = [
        SystemMessage(content="你可以调用工具来回答用户问题，每次只调用一个工具"),
        HumanMessage(content="计算 35 * 42 + 100 的结果"),
    ]
    
    print("\n=== Agent 工具调用循环 ===")
    max_iterations = 3
    
    for i in range(max_iterations):
        response = llm.invoke(messages, tools=registry)
        
        if response.has_tool_calls:
            tc = response.tool_calls[0]
            print(f"第{i+1}轮 LLM 调用工具: {tc.function_name}({tc.arguments})")
            
            # 将 AI 的 tool_call 消息加入历史
            messages.append(AIMessage(
                content=response.content,
                tool_calls=response.tool_calls
            ))
            
            # 执行工具并加入结果
            result = registry.execute(tc)
            messages.append(result)
            print(f"  → 工具返回: {result.content}")
        else:
            print(f"第{i+1}轮 最终回答: {response.content}")
            break
    else:
        print("达到最大迭代次数，停止循环")


def test_manual_tool_invoke():
    """测试手动调用工具"""
    print("\n=== 手动调用工具 ===")
    
    # 直接执行
    tool = registry.get("get_weather")
    result = tool.run(city="上海")
    print(f"get_weather(上海) = {result}")
    
    # 通过 ToolCall 执行
    from tiny_agent.core.base import ToolCall
    tc = ToolCall(id="call_001", function_name="calculator", arguments={"expression": "2 ** 10"})
    msg = registry.execute(tc)
    print(f"calculator(2**10) = {msg.content}")


if __name__ == "__main__":
    test_tool_registry()
    test_tool_call_with_llm()
    test_agent_loop()
    test_manual_tool_invoke()
