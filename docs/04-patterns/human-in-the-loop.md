# Human-in-the-Loop 패턴

> 중요한 작업 전에 사용자 승인을 요청하는 패턴입니다.

## 개요

Human-in-the-Loop (HITL) 패턴은 에이전트가 특정 도구를 실행하기 전에 사용자 승인을 요청합니다. 파일 수정, 외부 API 호출, 코드 실행 등 위험한 작업에 안전장치를 제공합니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Human-in-the-Loop Flow                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────┐                                                   │
│   │   Agent     │                                                   │
│   └──────┬──────┘                                                   │
│          │                                                          │
│          ▼                                                          │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  도구 호출: edit_file("/src/main.py", ...)                   │  │
│   └─────────────────────────────────────────────────────────────┘  │
│          │                                                          │
│          ▼                                                          │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  HumanInTheLoopMiddleware                                    │  │
│   │  ──────────────────────────────────────────────────────────  │  │
│   │  interrupt_on: {"edit_file": True}                           │  │
│   │  → 인터럽트 발생!                                            │  │
│   └─────────────────────────────────────────────────────────────┘  │
│          │                                                          │
│          ▼                                                          │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  ⏸️ 실행 일시 중지                                           │  │
│   │                                                              │  │
│   │  사용자에게 표시:                                            │  │
│   │  "edit_file('/src/main.py', old='...', new='...')            │  │
│   │   승인하시겠습니까? [Y/n]"                                   │  │
│   └─────────────────────────────────────────────────────────────┘  │
│          │                                                          │
│     ┌────┴────┐                                                     │
│     │         │                                                     │
│     ▼         ▼                                                     │
│   승인 ✅    거부 ❌                                                │
│     │         │                                                     │
│     ▼         ▼                                                     │
│   실행      피드백                                                  │
│   계속      제공                                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 기본 설정

### interrupt_on 파라미터

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    interrupt_on={
        "edit_file": True,      # 모든 파일 편집 전 승인
        "write_file": True,     # 모든 파일 생성 전 승인
        "execute": True,        # 모든 명령 실행 전 승인
    },
)
```

### 조건부 인터럽트

```python
from langchain.agents.middleware import InterruptOnConfig

agent = create_deep_agent(
    interrupt_on={
        # 특정 경로만 인터럽트
        "edit_file": InterruptOnConfig(
            condition=lambda args: "/src/" in args.get("file_path", "")
        ),
        # 특정 명령만 인터럽트
        "execute": InterruptOnConfig(
            condition=lambda args: "rm" in args.get("command", "")
        ),
    },
)
```

---

## InterruptOnConfig

### 구조

```python
class InterruptOnConfig(TypedDict):
    """인터럽트 설정"""

    condition: Callable[[dict], bool] | None
    """인터럽트 조건 함수. True 반환 시 인터럽트"""

    message: str | None
    """사용자에게 표시할 커스텀 메시지"""
```

### 사용 예시

```python
# 조건 함수 정의
def should_interrupt_edit(args: dict) -> bool:
    """편집 대상이 민감한 파일인지 확인"""
    sensitive_paths = ["/config/", "/secrets/", "/.env"]
    file_path = args.get("file_path", "")
    return any(p in file_path for p in sensitive_paths)

# 적용
agent = create_deep_agent(
    interrupt_on={
        "edit_file": InterruptOnConfig(
            condition=should_interrupt_edit,
            message="민감한 파일을 수정하려고 합니다. 승인하시겠습니까?",
        ),
    },
)
```

---

## 체크포인터 필수

HITL은 체크포인터가 필수입니다. 상태를 저장해야 재개가 가능합니다.

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_deep_agent(
    checkpointer=MemorySaver(),  # 필수!
    interrupt_on={
        "edit_file": True,
    },
)
```

---

## 실행 및 재개

### 기본 흐름

```python
from langgraph.checkpoint.memory import MemorySaver

# 1. 에이전트 생성 (체크포인터 필수)
agent = create_deep_agent(
    checkpointer=MemorySaver(),
    interrupt_on={"edit_file": True},
)

# 2. 초기 실행
config = {"configurable": {"thread_id": "session-1"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "main.py를 수정해주세요"}]},
    config=config,
)

# 3. 인터럽트 확인
if result.get("__interrupt__"):
    print("인터럽트 발생!")
    print(f"도구: {result['__interrupt__']['tool_name']}")
    print(f"인자: {result['__interrupt__']['tool_args']}")

    # 4. 사용자 승인
    approval = input("승인하시겠습니까? [Y/n]: ")

    if approval.lower() != 'n':
        # 5. 승인 후 재개
        result = agent.invoke(
            None,  # 새 메시지 없이 재개
            config=config,
        )
    else:
        # 6. 거부 시 피드백 제공
        result = agent.invoke(
            {"messages": [{
                "role": "user",
                "content": "이 수정은 승인하지 않습니다. 다른 방법을 제안해주세요."
            }]},
            config=config,
        )
```

### 비동기 실행

```python
async def run_with_approval(agent, messages, config):
    """인터럽트를 처리하며 에이전트 실행"""
    result = await agent.ainvoke(
        {"messages": messages},
        config=config,
    )

    while result.get("__interrupt__"):
        interrupt = result["__interrupt__"]
        print(f"\n⚠️ 승인 필요: {interrupt['tool_name']}")
        print(f"인자: {interrupt['tool_args']}")

        approval = input("승인? [Y/n]: ")

        if approval.lower() == 'n':
            # 거부
            result = await agent.ainvoke(
                {"messages": [{
                    "role": "user",
                    "content": f"{interrupt['tool_name']} 실행을 거부합니다."
                }]},
                config=config,
            )
        else:
            # 승인
            result = await agent.ainvoke(None, config=config)

    return result
```

---

## 실용 예제

### SQL 쿼리 승인

```python
@tool
def execute_sql(query: str) -> str:
    """SQL 쿼리를 실행합니다."""
    # 실제 구현
    pass

def should_approve_sql(args: dict) -> bool:
    """위험한 SQL 명령 감지"""
    dangerous = ["DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT"]
    query = args.get("query", "").upper()
    return any(cmd in query for cmd in dangerous)

agent = create_deep_agent(
    tools=[execute_sql],
    checkpointer=MemorySaver(),
    interrupt_on={
        "execute_sql": InterruptOnConfig(
            condition=should_approve_sql,
            message="데이터 변경 쿼리입니다. 실행하시겠습니까?",
        ),
    },
)
```

### 외부 API 호출 승인

```python
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """이메일을 발송합니다."""
    # 실제 구현
    pass

agent = create_deep_agent(
    tools=[send_email],
    checkpointer=MemorySaver(),
    interrupt_on={
        "send_email": True,  # 모든 이메일 발송 전 승인
    },
)
```

### 프로덕션 배포 승인

```python
@tool
def deploy_to_production(service: str, version: str) -> str:
    """프로덕션에 배포합니다."""
    # 실제 구현
    pass

agent = create_deep_agent(
    tools=[deploy_to_production],
    checkpointer=MemorySaver(),
    interrupt_on={
        "deploy_to_production": InterruptOnConfig(
            message="⚠️ 프로덕션 배포입니다! 정말 진행하시겠습니까?",
        ),
    },
)
```

---

## 서브에이전트와 HITL

서브에이전트에도 개별적으로 HITL을 설정할 수 있습니다.

```python
# 서브에이전트 정의
code_modifier = {
    "name": "code-modifier",
    "description": "코드를 수정하는 에이전트",
    "system_prompt": "당신은 코드 수정 전문가입니다.",
    "interrupt_on": {
        "edit_file": True,  # 서브에이전트의 파일 편집도 승인 필요
    },
}

# 메인 에이전트
agent = create_deep_agent(
    subagents=[code_modifier],
    checkpointer=MemorySaver(),
    # 메인 에이전트는 인터럽트 없음
)
```

---

## 모범 사례

### 1. 적절한 범위 설정

```python
# ✅ 좋은 예: 특정 조건에서만 인터럽트
interrupt_on={
    "edit_file": InterruptOnConfig(
        condition=lambda args: "/prod/" in args.get("file_path", "")
    ),
}

# ❌ 나쁜 예: 모든 작업 인터럽트 (사용자 피로)
interrupt_on={
    "edit_file": True,
    "write_file": True,
    "read_file": True,  # 읽기까지 인터럽트 = 과도함
}
```

### 2. 명확한 메시지 제공

```python
# ✅ 좋은 예: 구체적인 정보 제공
InterruptOnConfig(
    message="프로덕션 데이터베이스에서 DELETE 쿼리를 실행합니다. "
            "영향받는 행 수: 약 1,000건"
)

# ❌ 나쁜 예: 모호한 메시지
InterruptOnConfig(message="계속하시겠습니까?")
```

### 3. 일괄 승인 고려

```python
# 여러 파일 수정 시 일괄 승인 옵션
def batch_approval_flow(agent, files_to_modify):
    """여러 파일 수정을 일괄 승인"""
    print(f"다음 {len(files_to_modify)}개 파일을 수정합니다:")
    for f in files_to_modify:
        print(f"  - {f}")

    approval = input("전체 승인? [Y/n/개별]: ")

    if approval == "개별":
        # 개별 승인 모드
        return "individual"
    elif approval.lower() != 'n':
        return "all"
    else:
        return "none"
```

---

## 문제 해결

### 체크포인터 없이 실행

```
Error: Checkpointer required for interrupt handling
```

**해결**: 체크포인터 추가

```python
agent = create_deep_agent(
    checkpointer=MemorySaver(),
    interrupt_on={...},
)
```

### 재개가 안 됨

**확인 사항**:
1. 동일한 `thread_id` 사용
2. 동일한 체크포인터 인스턴스
3. 상태가 저장되었는지 확인

```python
# thread_id 일관성 유지
config = {"configurable": {"thread_id": "same-id"}}
```

### 인터럽트 후 상태 손실

```python
# 프로덕션용: 영속 체크포인터 사용
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@host:5432/db"
)
```

---

## 다음 단계

- [커스텀 도구 패턴](./custom-tools.md)
- [병렬 서브에이전트 패턴](./parallel-subagents.md)
- [상태 관리](../01-architecture/state-management.md)
