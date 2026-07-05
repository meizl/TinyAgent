"""
RAG 检索增强生成测试

注意：DeepSeek 不支持 Embedding API，RAG 部分需要支持 Embedding 的端点。
切分器测试无需 API，可直接运行。
"""

from tiny_agent.rag import (
    Retriever,
    OpenAIEmbeddings,
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    ParagraphTextSplitter,
)


def test_custom_splitter():
    """测试自定义切分器（无需 API）"""
    print("=== 文本切分器测试 ===\n")

    text = (
        "第一段：人工智能正在改变世界。\n\n"
        "第二段：机器学习是 AI 的核心技术。\n\n"
        "第三段：深度学习使用神经网络进行特征提取。\n\n"
        "第四段：自然语言处理让计算机理解人类语言。\n\n"
        "第五段：计算机视觉使机器能够识别图像内容。"
    )

    # 段落切分
    p = ParagraphTextSplitter()
    chunks = p.split_text(text)
    print(f"段落切分: {len(chunks)} 块")
    for i, c in enumerate(chunks):
        print(f"  块{i+1}: {c[:40]}...")

    # 字符切分
    print()
    c = CharacterTextSplitter(chunk_size=30, chunk_overlap=5)
    chunks = c.split_text(text)
    print(f"字符切分(chunk_size=30): {len(chunks)} 块")
    for i, c in enumerate(chunks):
        print(f"  块{i+1}: {c[:40]}...")

    # 递归切分
    print()
    r = RecursiveCharacterTextSplitter(chunk_size=100)
    chunks = r.split_text(text)
    print(f"递归切分(chunk_size=100): {len(chunks)} 块")
    for i, c in enumerate(chunks):
        print(f"  块{i+1}: {c[:40]}...")


def test_loader_formats():
    """测试支持的文档格式"""
    print("\n=== 支持的文档格式 ===")
    from tiny_agent.rag.loaders import _LOADER_MAP
    for ext, loader in _LOADER_MAP.items():
        print(f"  {ext}: {loader.__name__}")


if __name__ == "__main__":
    test_custom_splitter()
    test_loader_formats()
