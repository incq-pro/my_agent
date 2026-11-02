from enum import Enum
from typing import Callable

from jsonschema import validate
from tools import RiskLevel, Tool, ToolRegistry


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


# 审批回调由可信的命令行入口提供
ApprovalCallback = Callable[[Tool, dict], bool]


class PermissionGate:
    """
    Agent 工具执行权限网关。

    决策顺序：
    1. 强制拒绝
    2. 需要人工审批
    3. 自动允许
    """

    def decide(
        self,
        tool: Tool,
        arguments: dict,
    ) -> PermissionDecision:

        # ---------- 第一关：硬拒绝 ----------

        if tool.name == "restart_service":
            environment = arguments["environment"]

            if environment == "production":
                return PermissionDecision.DENY

        # ---------- 第二关：需要用户审批 ----------

        if tool.risk in {
            RiskLevel.WRITE,
            RiskLevel.DANGEROUS,
        }:
            return PermissionDecision.ASK

        # ---------- 第三关：安全只读操作 ----------

        if tool.risk == RiskLevel.READ:
            return PermissionDecision.ALLOW

        # 未识别的风险类型，默认拒绝
        return PermissionDecision.DENY

    def execute(
        self,
        registry: ToolRegistry,
        tool_name: str,
        arguments: dict,
        approval_callback: ApprovalCallback | None = None,
    ):
        # 1. 工具白名单
        # 只有 Registry 中注册的工具可以执行
        tool = registry.get(tool_name)

        # 2. 参数校验
        # 不能信任模型返回的参数
        validate(
            instance=arguments,
            schema=tool.parameters,
        )

        # 3. 权限判断
        decision = self.decide(tool, arguments)

        if decision == PermissionDecision.DENY:
            raise PermissionError(f"权限拒绝: {tool_name} {arguments}")

        if decision == PermissionDecision.ASK:
            # 默认拒绝，不能由模型自行批准
            approved = (
                approval_callback(tool, arguments)
                if approval_callback is not None
                else False
            )

            if not approved:
                raise PermissionError(f"用户未批准: {tool_name}")

        # 4. 所有检查通过后才真正执行
        return registry.execute(
            tool_name,
            arguments,
        )
