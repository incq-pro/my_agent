import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

client = OpenAI()
MODEL = os.environ["OPENAI_MODEL"]


class OrderDiagnosis(BaseModel):
    reason: Literal[
        "payment_pending",
        "out_of_stock",
        "logistics_delay",
        "unknown",
    ] = Field(description="订单未发货的原因类别")

    evidence: list[str] = Field(description="支持诊断结论的已知证据")

    suggested_action: Literal[
        "check_payment",
        "check_inventory",
        "contact_logistics",
        "request_more_information",
    ]

    requires_approval: bool

    confidence: float = Field(ge=0.0, le=1.0)


SCHEMA_HINT = json.dumps(OrderDiagnosis.model_json_schema(), ensure_ascii=False)


def diagnose_order(order_info: str) -> OrderDiagnosis:
    response = client.responses.parse(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": f"""
你是订单故障诊断助手。

根据提供的证据判断订单未发货的可能原因。
不得编造未提供的事实。
证据不足时，将 reason 设置为 unknown，
将 suggested_action 设置为 request_more_information。
只提供诊断和建议，不要声称已经执行操作。
涉及退款、取消订单或修改订单的操作必须人工审批。

只输出一个 JSON 对象，必须包含下面 schema 中的全部字段（evidence、requires_approval、confidence 也不能省略）：
{SCHEMA_HINT}
""",
            },
            {
                "role": "user",
                "content": order_info,
            },
        ],
        text_format=OrderDiagnosis,
    )

    result = response.output_parsed

    if result is None:
        raise RuntimeError("模型没有返回可解析的诊断结果")

    return result


if __name__ == "__main__":
    order_info = """
    订单号：ORD-1024
    支付状态：已支付
    库存状态：库存充足
    物流状态：未创建物流单
    """

    diagnosis = diagnose_order(order_info)

    print(diagnosis)
    print(diagnosis.model_dump_json(indent=2))
