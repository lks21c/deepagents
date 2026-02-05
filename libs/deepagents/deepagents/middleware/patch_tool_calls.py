"""
모듈명: patch_tool_calls.py
설명: 메시지 히스토리에서 미완료된(dangling) 도구 호출을 패치하는 미들웨어

이 미들웨어는 AIMessage에 포함된 도구 호출(tool_call) 중 대응하는
ToolMessage가 없는 경우를 감지하고, 자동으로 취소 메시지를 생성하여
메시지 히스토리의 일관성을 유지합니다.

문제 상황:
- AI가 도구 호출을 요청했으나, 다른 메시지가 먼저 도착하여 도구 실행이 취소됨
- 이로 인해 tool_call_id에 대응하는 ToolMessage가 없는 "dangling" 상태 발생
- 대부분의 LLM API는 tool_call과 ToolMessage의 쌍을 기대하므로 오류 발생 가능

해결 방법:
- 에이전트 실행 전에 메시지 히스토리를 스캔하여 dangling tool call 감지
- 감지된 각 dangling tool call에 대해 취소 메시지를 자동 생성하여 삽입

주요 클래스:
- PatchToolCallsMiddleware: 미완료 도구 호출 패치 미들웨어
"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite


class PatchToolCallsMiddleware(AgentMiddleware):
    """
    메시지 히스토리에서 미완료된(dangling) 도구 호출을 패치하는 미들웨어

    이 미들웨어는 에이전트 실행 전에 메시지 히스토리를 검사하여,
    AIMessage의 tool_call에 대응하는 ToolMessage가 없는 경우를 감지합니다.
    감지된 dangling tool call에 대해 자동으로 취소 ToolMessage를 생성하여
    메시지 히스토리의 일관성을 유지합니다.

    사용 시나리오:
    1. 사용자가 도구 호출 중간에 새 메시지를 보낸 경우
    2. 네트워크 오류로 도구 실행이 중단된 경우
    3. 타임아웃으로 도구 응답이 누락된 경우

    예시:
        ```python
        from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
        from langchain.agents import create_agent

        agent = create_agent(
            "openai:gpt-4o",
            middleware=[PatchToolCallsMiddleware()],
        )
        ```
    """

    def before_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """
        에이전트 실행 전 dangling tool call 처리

        모든 AIMessage의 tool_call을 검사하여 대응하는 ToolMessage가
        없는 경우 취소 메시지를 자동 생성합니다.

        인자:
            state: 현재 에이전트 상태 (messages 키 포함)
            runtime: 런타임 컨텍스트 (사용하지 않음)

        반환값:
            dict[str, Any] | None: 패치된 메시지 리스트를 포함한 상태 업데이트,
                                   패치가 필요 없으면 None
        """
        messages = state["messages"]
        # 메시지가 없거나 빈 경우 처리할 것이 없음
        if not messages or len(messages) == 0:
            return None

        patched_messages = []
        # 메시지를 순회하며 dangling tool call 감지 및 패치
        for i, msg in enumerate(messages):
            patched_messages.append(msg)
            # AIMessage이고 tool_calls가 있는 경우 검사
            if msg.type == "ai" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # 현재 메시지 이후에 대응하는 ToolMessage가 있는지 확인
                    corresponding_tool_msg = next(
                        (msg for msg in messages[i:] if msg.type == "tool" and msg.tool_call_id == tool_call["id"]),
                        None,
                    )
                    if corresponding_tool_msg is None:
                        # Dangling tool call 발견 - 취소 ToolMessage 생성
                        tool_msg = (
                            f"Tool call {tool_call['name']} with id {tool_call['id']} was "
                            "cancelled - another message came in before it could be completed."
                        )
                        patched_messages.append(
                            ToolMessage(
                                content=tool_msg,
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )

        # Overwrite를 사용하여 전체 메시지 리스트를 교체
        return {"messages": Overwrite(patched_messages)}
