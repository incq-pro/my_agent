"""
@author zhangqiang
@date 2026/06/11
通过 OpenAI SDK 调用 MiniMax 模型
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 环境变量
load_dotenv()

MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")


def create_minimax_client() -> OpenAI:
    """创建 MiniMax 的 OpenAI 兼容客户端"""
    if not MINIMAX_API_KEY:
        raise ValueError("环境变量 MINIMAX_API_KEY 未设置，请在 .env 文件中配置")

    return OpenAI(
        base_url=MINIMAX_BASE_URL,
        api_key=MINIMAX_API_KEY,
    )


def chat(messages: list[dict], model: str | None = None) -> str:
    """
    调用 MiniMax 模型进行对话

    :param messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
    :param model: 模型名称，默认使用环境变量中的配置
    :return: 模型回复内容
    """
    client = create_minimax_client()
    response = client.chat.completions.create(
        model=model or MINIMAX_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content


def main():
    """示例：调用 MiniMax 模型进行对话"""
    print("正在通过 OpenAI SDK 调用 MiniMax 模型...")

    messages = [
        {"role": "system", "content": "你是一个幽默的技术导师。"},
        {"role": "user", "content": "一句话解释为什么大模型是无状态（Stateless）的？"},
    ]

    reply = chat(messages)
    print("\nMiniMax 导师的回复：")
    print("=" * 30)
    print(reply)


if __name__ == "__main__":
    main()
