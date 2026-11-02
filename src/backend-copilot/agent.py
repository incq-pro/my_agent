
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from security import PermissionGate
from tools import ToolRegistry

from openai import OpenAI

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL")

if not MODEL:
    raise RuntimeError("请在 .env 中配置 OPENAI_MODEL")


# ---------- 结构化程度较高的运行日志 ----------

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)s "
        "%(name)s %(message)s"
    ),
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            "logs/agent.log",
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger("backend_copilot")


class Agent:
    def __init__(
        self,
        registry: ToolRegistry,
        client=None,
        model: str = MODEL,
        max_turns: int = 8,
        approval_callback=None,
    ):
        if max_turns < 1:
            raise ValueError("max_turns 必须大于等于 1")

        self.registry = registry
        self.client = client or OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = model
        self.max_turns = max_turns
        self.gate = PermissionGate()
        self.approval_callback = approval_callback

    def run(self, user_message: str) -> str:
        run_id = f"run-{time.time_ns()}"

        input_items = [
            {
                "role": "system",
                "content": (
                    "你是后端研发与运维助手。"
                    "使用工具获取证据后再分析。"
                    "不要编造工具没有返回的数据。"
                    "证据不足时明确说明不确定性。"
                ),
            },
            {"role": "user", "content": user_message},
        ]

        logger.info(
            "event=run_start run_id=%s",
            run_id,
        )

        try:
            for turn in range(1, self.max_turns + 1):
                logger.info(
                    "event=model_request run_id=%s turn=%s",
                    run_id,
                    turn,
                )

                # 模型请求失败时交由外层处理
                response = self.client.responses.create(
                    model=self.model,
                    input=input_items,
                    tools=self.registry.get_model_schemas(),
                    parallel_tool_calls=False,
                )

                # 保留本轮模型输出，维持工具调用上下文
                input_items += response.output

                tool_calls = [
                    item
                    for item in response.output
                    if item.type == "function_call"
                ]

                if not tool_calls:
                    logger.info(
                        "event=run_complete run_id=%s turns=%s",
                        run_id,
                        turn,
                    )
                    return response.output_text

                for call in tool_calls:
                    started = time.perf_counter()

                    logger.info(
                        "event=tool_start run_id=%s tool=%s",
                        run_id,
                        call.name,
                    )

                    try:
                        arguments = json.loads(call.arguments)

                        if not isinstance(arguments, dict):
                            raise ValueError(
                                "工具参数必须是 JSON 对象"
                            )

                        result = self.gate.execute(
                            registry=self.registry,
                            tool_name=call.name,
                            arguments=arguments,
                            approval_callback=self.approval_callback,
                        )

                        output = {
                            "ok": True,
                            "data": result,
                        }

                    except Exception as exc:
                        # 记录异常类型，向模型反馈可处理的错误
                        logger.warning(
                            "event=tool_error run_id=%s "
                            "tool=%s error_type=%s",
                            run_id,
                            call.name,
                            type(exc).__name__,
                        )

                        output = {
                            "ok": False,
                            "error": (
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }

                    elapsed_ms = round(
                        (time.perf_counter() - started) * 1000,
                        2,
                    )

                    logger.info(
                        "event=tool_end run_id=%s "
                        "tool=%s elapsed_ms=%s ok=%s",
                        run_id,
                        call.name,
                        elapsed_ms,
                        output["ok"],
                    )

                    input_items.append({
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(
                            output,
                            ensure_ascii=False,
                        ),
                    })

            logger.warning(
                "event=max_turns_reached run_id=%s max_turns=%s",
                run_id,
                self.max_turns,
            )

            return (
                f"任务未在 {self.max_turns} 轮内完成，"
                "已停止执行。"
            )

        except Exception:
            logger.exception(
                "event=run_failed run_id=%s",
                run_id,
            )
            raise

        finally:
            logger.info(
                "event=run_end run_id=%s",
                run_id,
            )
