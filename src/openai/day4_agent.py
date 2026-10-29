import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()
MODEL = os.environ["OPENAI_MODEL"]

# 模拟订单服务返回的数据
ORDERS = {
    "ORD-1001": {
        "order_id": "ORD-1001",
        "payment_status": "paid",
        "inventory_status": "in_stock",
        "shipping_status": "pending",
    },
    "ORD-1002": {
        "order_id": "ORD-1002",
        "payment_status": "unpaid",
        "inventory_status": "in_stock",
        "shipping_status": "pending",
    },
}


# 这是应用程序实际执行的 Python 函数
def get_order(order_id: str) -> dict:
    order = ORDERS.get(order_id)

    if order is None:
        return {
            "found": False,
            "order_id": order_id,
        }

    return {
        "found": True,
        **order,
    }


# 这是提供给模型的工具说明
TOOLS = [
    {
        "type": "function",
        "name": "get_order",
        "description": "查询订单的支付、库存和发货状态",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "需要查询的订单 ID",
                }
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


def execute_tool(name: str, arguments: str) -> str:
    """根据模型的工具请求，执行允许调用的函数。"""

    if name != "get_order":
        return json.dumps(
            {"error": "Unsupported tool"},
            ensure_ascii=False,
        )

    try:
        args = json.loads(arguments)
        result = get_order(args["order_id"])
        return json.dumps(result, ensure_ascii=False)
    except (json.JSONDecodeError, KeyError, TypeError):
        return json.dumps(
            {"error": "Invalid tool arguments"},
            ensure_ascii=False,
        )


def ask_agent(question: str) -> str:
    instructions = (
        "你是订单排查助手。"
        "需要订单事实时，先调用 get_order 查询。"
        "只能根据工具返回的事实解释订单状态；"
        "如果订单不存在，要明确告知用户。"
    )
    # 兼容网关不保存服务端状态，由客户端自己维护完整的对话历史
    input_items = [{"role": "user", "content": question}]

    # 模型可能先请求工具，收到结果后再请求其他工具。
    # 此示例只有一个工具，因此通常只需执行一轮。
    for i in range(5):
        response = client.responses.create(
            model=MODEL,
            instructions=instructions,
            input=input_items,
            tools=TOOLS,
        )
        print(i, response.output)

        function_calls = [item for item in response.output if item.type == "function_call"]

        if not function_calls:
            return response.output_text

        # 先把模型的工具调用请求放回历史，再追加工具执行结果
        input_items.extend(response.output)
        
        for call in function_calls:
            output = execute_tool(
                call.name,
                call.arguments,
            )

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": output,
                }
            )

    raise RuntimeError("工具调用超过最大轮数")


if __name__ == "__main__":
    question = "订单 ORD-1001 为什么还没发货？"
    print(ask_agent(question))
