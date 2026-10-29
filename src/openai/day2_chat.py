import os
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    OpenAI,
    RateLimitError,
)

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL")
if not api_key:
    raise RuntimeError("请在 .env 中配置 OPENAI_API_KEY")

if not model:
    raise RuntimeError("请在 .env 中配置 OPENAI_MODEL")

client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))

SYSTEM_PROMPT = """
你是一名资深 Java 后端工程师。
回答问题时，优先给出清晰、准确、可执行的建议。
如果信息不足，请明确指出，不要编造事实。
"""


def ask_llm(question: str) -> str:
    """调用大模型并返回文本回答"""
    if not question.strip():
        raise ValueError("问题不能为空")
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=question,
    )
    return response.output_text


if __name__ == "__main__":
    history: list[dict[str, str]] = []
    print("后端研发助手已启动，输入 exit 退出。")

    while True:
        question = input("\n你：").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        history.append(
            {
                "role": "user",
                "content": question,
            }
        )
        try:
            response = client.responses.create(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=history,
            )
            answer = response.output_text
            print(f"\n助手：{answer}")
            # 保存助手回答，使下一轮请求能够读取上下文。
            history.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )
            if response.usage:
                print(
                    f"\n[本次请求 Token 用量] 输入：{response.usage.input_tokens}，输出：{response.usage.output_tokens}"
                )
        except (RateLimitError, APIConnectionError, APIStatusError) as exc:
            print(f"调用失败：{type(exc).__name__}")
            # 请求失败时移除本轮未完成的问题。
            history.pop()
