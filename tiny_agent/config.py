"""
TinyAgent 配置管理

从 .env 文件和环境变量读取配置，优先级：环境变量 > .env 文件。
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# 尝试加载项目根目录下的 .env 文件
try:
    from dotenv import load_dotenv
    # 向上查找项目根目录（包含 .env 文件的目录）
    _project_root = Path(__file__).parent.parent
    _env_file = _project_root / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
    else:
        # 兜底：尝试从当前工作目录加载
        load_dotenv()
except ImportError:
    pass


@dataclass
class Config:
    """TinyAgent 全局配置"""

    # OpenAI / 兼容 API 配置
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL") or None)
    default_model: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo"))


# 全局配置实例
config = Config()
