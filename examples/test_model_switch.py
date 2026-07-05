"""
模型切换测试示例

使用 LangChain 风格：不同模型使用不同实例，实例不可变。
"""

from tiny_agent import OpenAIAdapter, SystemMessage, HumanMessage

messages = [
    SystemMessage(content="你是一个友好的助手"),
    HumanMessage(content="用一句话介绍你自己")
]

# 模型 A：deepseek-v4-pro
llm_pro = OpenAIAdapter(model="deepseek-v4-pro")
print(f"[{llm_pro.model}] {llm_pro.invoke(messages).content}")

# 模型 B：deepseek-chat（新建实例，更轻快的模型）
llm_chat = OpenAIAdapter(model="deepseek-chat")
print(f"[{llm_chat.model}] {llm_chat.invoke(messages).content}")

# 模型 C：控制参数（低温 + 严格输出）
llm_strict = OpenAIAdapter(model="deepseek-v4-pro", temperature=0.0, max_tokens=30)
response = llm_strict.invoke(messages)
print(f"[{llm_strict.model} t=0.0 max=30] {response.content}")
