"""
模型参数测试示例

演示如何在初始化时设置默认参数，以及在调用时覆盖参数。
"""

from tiny_agent import OpenAIAdapter, SystemMessage, HumanMessage


def test_params():
    # 方式1：初始化时设置默认参数（低温，更稳定）
    llm = OpenAIAdapter(
        temperature=0.3,       # 低温，输出更确定
        presence_penalty=0.3,  # 轻微减少重复话题
    )
    
    messages = [
        SystemMessage(content="你是一个友好的助手"),
        HumanMessage(content="你好，告诉我一个有趣的事实")
    ]
    
    # 使用默认参数调用
    response = llm.invoke(messages)
    print("=== 默认参数 (temperature=0.3) ===")
    print(f"响应: {response.content}\n")
    
    # 方式2：调用时覆盖参数（高温，更随机有创意）
    response2 = llm.invoke(messages, temperature=0.9)
    print("=== 调用时覆盖 (temperature=0.9) ===")
    print(f"响应: {response2.content}\n")
    
    # 方式3：限制输出长度
    response3 = llm.invoke(messages, max_tokens=50)
    print("=== 限制长度 (max_tokens=50) ===")
    print(f"响应: {response3.content}\n")
    
    # 方式4：流式调用时覆盖参数
    print("=== 流式调用 (temperature=1.0) ===")
    for chunk in llm.stream(messages, temperature=1.0):
        print(chunk, end="", flush=True)
    print("\n")
    
    # 方式5：JSON 格式输出
    messages_json = [
        SystemMessage(content="你是一个数据助手，请始终用 JSON 格式回答"),
        HumanMessage(content="列出3种水果")
    ]
    response4 = llm.invoke(messages_json, response_format={"type": "json_object"})
    print("=== JSON 格式输出 ===")
    print(f"响应: {response4.content}")


if __name__ == "__main__":
    test_params()
