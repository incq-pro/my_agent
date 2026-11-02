from tools import build_registry
from agent import Agent


def ask_approval(tool, arguments):
    print("\n" + "=" * 45)
    print("⚠️ Agent 请求执行受限操作")
    print(f"工具：{tool.name}")
    print(f"风险：{tool.risk.value}")
    print(f"参数：{arguments}")
    print("=" * 45)

    try:
        answer = input("是否允许本次操作？[y/N]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False

    # 只有明确输入 y / yes 才批准
    return answer in {"y", "yes"}


def main():
    registry = build_registry()
    agent = Agent(
        registry=registry,
        approval_callback=ask_approval,
    )

    print("Backend Copilot Agent v0.1")
    print("输入 exit 或 quit 退出。\n")

    while True:
        try:
            question = input("你：").strip()

            if question.lower() in {"exit", "quit"}:
                print("再见！")
                break

            if not question:
                continue

            answer = agent.run(question)
            print(f"\nAgent：{answer}\n")

        except KeyboardInterrupt:
            print("\n已退出。")
            break

        except Exception as exc:
            print(
                f"\n本次任务执行失败：{type(exc).__name__}。"
                "请检查 logs/agent.log 获取详细信息。\n"
            )


if __name__ == "__main__":
    main()
