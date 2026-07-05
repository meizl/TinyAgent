"""
模型切换测试示例

演示如何动态切换模型，无需重建客户端。
"""

from tiny_agent import OpenAIAdapter, SystemMessage, HumanMessage


def test_model_switching():
    llm = OpenAIAdapter()
    
    messages = [
        SystemMessage(content="你是一个友好的助手"),
        HumanMessage(content="用一句话介绍你自己")
    ]
    
    print(f"当前模型: {llm.model}")
    response = llm.invoke(messages)
    print(f"响应: {response.content}\n")
    
    # 切换到另一个模型（如果配置支持）
    llm.set_model("deepseek-chat")
    print(f"切换后模型: {llm.model}")
    response2 = llm.invoke(messages)
    print(f"响应: {response2.content}\n")
    
    # 切回原模型
    llm.set_model("deepseek-v4-pro")
    print(f"切回模型: {llm.model}")


if __name__ == "__main__":
    test_model_switching()
