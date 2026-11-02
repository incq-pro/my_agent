from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from jsonschema import validate


class RiskLevel(str, Enum):
    READ = "read"
    WRITE = "write"
    DANGEROUS = "dangerous"


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Any]
    risk: RiskLevel

    def to_model_schema(self) -> dict:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": True,
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重复注册: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ValueError(f"未知工具: {name}")
        return self._tools[name]

    def get_model_schemas(self) -> list[dict]:
        return [tool.to_model_schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> Any:
        tool = self.get(name)

        # 服务端校验不能省略，不能完全信任模型输出
        validate(
            instance=arguments,
            schema=tool.parameters,
        )

        return tool.handler(**arguments)


# ---------- 业务工具：全部使用模拟数据 ----------


def get_service_status(service_name: str) -> dict:
    services = {
        "order-service": {
            "status": "abnormal",
            "error_rate": 0.35,
            "cpu_usage": 0.92,
        },
        "user-service": {
            "status": "healthy",
            "error_rate": 0.001,
            "cpu_usage": 0.30,
        },
    }

    return services.get(
        service_name,
        {"status": "unknown", "message": "服务不存在"},
    )


def get_recent_errors(service_name: str, limit: int) -> dict:
    logs = [
        "Database connection timeout",
        "Connection pool exhausted",
        "Request processing timeout",
        "Database connection timeout",
        "Connection pool exhausted",
    ]

    return {
        "service_name": service_name,
        "errors": logs[:limit],
    }


def search_docs(query: str) -> dict:
    docs = [
        {
            "title": "数据库连接池排障",
            "content": ("连接池耗尽时，应检查连接泄漏、连接池最大连接数和慢 SQL。"),
            "keywords": ["数据库", "连接池", "超时"],
        },
        {
            "title": "服务 CPU 过高排查",
            "content": "检查线程使用情况、热点代码和高频请求。",
            "keywords": ["CPU", "线程", "性能"],
        },
    ]

    query_lower = query.lower()
    results = [
        {
            "title": doc["title"],
            "content": doc["content"],
        }
        for doc in docs
        if any(keyword.lower() in query_lower for keyword in doc["keywords"])
    ]

    return {"query": query, "results": results}


def restart_service(
    service_name: str,
    environment: str,
) -> dict:
    """模拟服务重启，不执行真实重启操作。"""

    return {
        "service_name": service_name,
        "environment": environment,
        "status": "simulated_restart_success",
        "message": "模拟重启成功，未操作真实服务",
    }


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="get_service_status",
            description="查询服务状态、错误率和 CPU 使用率",
            parameters={
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                },
                "required": ["service_name"],
                "additionalProperties": False,
            },
            handler=get_service_status,
            risk=RiskLevel.READ,
        )
    )

    registry.register(
        Tool(
            name="get_recent_errors",
            description="查询指定服务最近的错误日志",
            parameters={
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                    },
                },
                "required": ["service_name", "limit"],
                "additionalProperties": False,
            },
            handler=get_recent_errors,
            risk=RiskLevel.READ,
        )
    )

    registry.register(
        Tool(
            name="search_docs",
            description="搜索后端故障排查文档",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=search_docs,
            risk=RiskLevel.READ,
        )
    )

    registry.register(
        Tool(
            name="restart_service",
            description="模拟重启指定环境的后端服务",
            parameters={
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                    },
                    "environment": {
                        "type": "string",
                        "enum": ["staging", "production"],
                    },
                },
                "required": [
                    "service_name",
                    "environment",
                ],
                "additionalProperties": False,
            },
            handler=restart_service,
            risk=RiskLevel.DANGEROUS,
        )
    )

    return registry
