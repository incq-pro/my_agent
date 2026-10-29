import os
import time

from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# 1. 加载环境变量
load_dotenv()
console = Console()


def create_client() -> Anthropic:
    """
    工厂函数：初始化 Anthropic 客户端。
    SDK 会自动读取环境变量中的 ANTHROPIC_API_KEY 和 ANTHROPIC_BASE_URL。
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    if not api_key:
        raise ValueError("未检测到 ANTHROPIC_API_KEY，请检查 .env 文件！")

    return Anthropic(
        api_key=api_key,
        base_url=base_url,  # 若为 None，SDK 默认请求 Anthropic 官方网关
        timeout=60.0,  # 后端铁律：任何外部 RPC 必须设置显式超时时间！
        max_retries=2,  # SDK 内置的基础网络层重试
    )


def inspect_messages_api():
    client = create_client()
    model_name = os.getenv("DEFAULT_MODEL", "MiniMax-M3")

    console.print(f"[bold cyan]🚀 正在通过 Anthropic SDK 调用模型:[/bold cyan] {model_name}")
    console.print(f"[bold cyan]🌐 当前网关地址 (Base URL):[/bold cyan] {client.base_url}\n")

    start_time = time.perf_counter()

    # 2. 发送标准的 Anthropic Messages 请求
    response = client.messages.create(
        model=model_name,
        max_tokens=2048,
        # 注意：Anthropic 协议中，system 是独立顶层参数，不在 messages 数组里！
        system="你是一位拥有10年经验的资深后端架构师，回答问题干脆利落，直击底层原理。",
        messages=[
            {"role": "user", "content": "请用三句话向后端工程师解释：为什么大语言模型本质上是无状态（Stateless）的？"}
        ],
        temperature=0.0,
        # 开启扩展思考（Extended Thinking），让模型先输出 ThinkingBlock 再输出 TextBlock
        # thinking={"type": "enabled", "budget_tokens": 1024},
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # 3. 深度解剖返回报文结构 (遍历多态 ContentBlock 数组)
    for idx, block in enumerate(response.content):
        if block.type == "thinking":
            console.print(
                Panel(block.thinking, title=f"🧠 Block [{idx}] - 模型内部推导流 (ThinkingBlock)", border_style="yellow")
            )
        elif block.type == "text":
            console.print(Panel(block.text, title=f"💬 Block [{idx}] - 最终正文输出 (TextBlock)", border_style="green"))
        elif block.type == "tool_use":
            console.print(
                Panel(
                    f"Tool: {block.name}\nInput: {block.input}",
                    title=f"🛠️ Block [{idx}] - 工具调用请求 (ToolUseBlock)",
                    border_style="magenta",
                )
            )

    # 4. 打印后端监控关键元数据（Stop Reason 与 Token 账单）
    console.print(
        f"\n[bold white]📊 调用审计元数据:[/bold white] "
        f"ID={response.id} | "
        f"StopReason=[bold yellow]{response.stop_reason}[/bold yellow] | "
        f"InputTokens={response.usage.input_tokens} | "
        f"OutputTokens={response.usage.output_tokens} | "
        f"Latency={elapsed_ms:.1f}ms"
    )


if __name__ == "__main__":
    inspect_messages_api()
