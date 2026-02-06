# 상태 관리 (State Management)

> LangGraph 기반의 에이전트 상태 관리 및 데이터 흐름을 설명합니다.

## 개요

Deep Agents는 LangGraph의 상태 관리 시스템을 기반으로 합니다. 상태는 TypedDict로 정의되며, 각 미들웨어가 자신의 상태 스키마를 확장합니다.

```mermaid
graph TD
    subgraph StateManagement["State Management Flow"]
        IS["Initial State<br/>{messages: [HumanMessage&#40;'질문...'&#41;]}"]
        ME["Middleware State Extensions<br/>TodoListMiddleware → + todos: list[Todo]<br/>MemoryMiddleware → + memory_contents: dict<br/>SkillsMiddleware → + skills_metadata: list[SkillMetadata]<br/>SubAgentMiddleware → &#40;no state extension&#41;"]
        FS["Final State<br/>{messages: [HumanMessage, AIMessage, ...],<br/>todos: [...],<br/>memory_contents: {...},<br/>skills_metadata: [...]}"]

        IS --> ME --> FS
    end
```

---

## 기본 상태 스키마

### AgentState

LangChain 에이전트의 기본 상태 스키마입니다.

```python
from typing import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """에이전트 기본 상태"""

    messages: Annotated[list[BaseMessage], add_messages]
    """대화 메시지 히스토리. add_messages 리듀서로 병합됨."""
```

**`add_messages` 리듀서**:
- 새 메시지를 기존 목록에 추가
- 동일 ID의 메시지는 덮어쓰기
- 순서 유지

---

## 미들웨어별 상태 확장

### TodoListMiddleware 상태

```python
class TodoState(AgentState):
    """할 일 목록 미들웨어 상태"""

    todos: NotRequired[Annotated[list[Todo], PrivateStateAttr]]
```

### MemoryMiddleware 상태

```python
class MemoryState(AgentState):
    """메모리 미들웨어 상태"""

    memory_contents: NotRequired[Annotated[dict[str, str], PrivateStateAttr]]
    """소스 경로 → 메모리 내용 매핑"""
```

### SkillsMiddleware 상태

```python
class SkillsState(AgentState):
    """스킬 미들웨어 상태"""

    skills_metadata: NotRequired[Annotated[list[SkillMetadata], PrivateStateAttr]]
    """로드된 스킬 메타데이터 목록"""
```

---

## PrivateStateAttr

`PrivateStateAttr`는 상태 필드를 "비공개"로 표시합니다.

```python
from langchain.agents.middleware.types import PrivateStateAttr

class MyState(AgentState):
    internal_data: NotRequired[Annotated[dict, PrivateStateAttr]]
```

**특성**:
- 서브에이전트에게 전파되지 않음
- 부모 에이전트로 반환되지 않음
- 상태 직렬화 시 제외될 수 있음

**사용 사례**:
- 미들웨어 내부 데이터
- 컨텍스트 크기 최적화
- 중간 계산 결과

---

## 상태 업데이트 패턴

### before_agent 훅에서 상태 업데이트

```python
class MyMiddleware(AgentMiddleware):
    state_schema = MyState

    def before_agent(
        self,
        state: MyState,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> MyStateUpdate | None:
        """에이전트 실행 전 상태 업데이트"""

        # 이미 로드되었으면 건너뛰기
        if "my_data" in state:
            return None

        # 데이터 로드
        data = self._load_data()

        # 상태 업데이트 반환
        return MyStateUpdate(my_data=data)
```

### 도구 호출에서 상태 업데이트

```python
from langgraph.types import Command

@tool
def my_tool(runtime: ToolRuntime, arg: str) -> Command:
    """상태를 업데이트하는 도구"""

    # 작업 수행
    result = process(arg)

    # Command로 상태 업데이트 + 메시지 반환
    return Command(
        update={
            "my_data": result,
            "messages": [ToolMessage(content="완료", tool_call_id=runtime.tool_call_id)],
        }
    )
```

---

## 메시지 타입

### 주요 메시지 클래스

```python
from langchain_core.messages import (
    HumanMessage,     # 사용자 입력
    AIMessage,        # 에이전트 응답
    SystemMessage,    # 시스템 프롬프트
    ToolMessage,      # 도구 실행 결과
)
```

### 메시지 구조

```python
# HumanMessage
HumanMessage(content="안녕하세요!")

# AIMessage with tool calls
AIMessage(
    content="파일을 읽겠습니다.",
    tool_calls=[
        {
            "id": "call_123",
            "name": "read_file",
            "args": {"file_path": "/src/main.py"},
        }
    ],
)

# ToolMessage
ToolMessage(
    content="파일 내용...",
    tool_call_id="call_123",
)

# SystemMessage
SystemMessage(content="당신은 Python 전문가입니다.")
```

---

## 체크포인터 (Checkpointer)

체크포인터는 에이전트 상태를 영속화합니다.

### 메모리 체크포인터

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_deep_agent(
    checkpointer=MemorySaver(),
)

# 세션 ID로 대화 지속
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "제 이름은 김철수입니다"}]},
    config={"configurable": {"thread_id": "session-1"}}
)

result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "제 이름이 뭐였죠?"}]},
    config={"configurable": {"thread_id": "session-1"}}  # 같은 세션
)
# → "김철수님이라고 하셨습니다."
```

### SQLite 체크포인터

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 파일 기반 영속화
agent = create_deep_agent(
    checkpointer=SqliteSaver.from_conn_string("./checkpoints.db"),
)
```

### PostgreSQL 체크포인터

```python
from langgraph.checkpoint.postgres import PostgresSaver

# 프로덕션용 체크포인터
agent = create_deep_agent(
    checkpointer=PostgresSaver.from_conn_string(
        "postgresql://user:pass@host:5432/db"
    ),
)
```

---

## 상태 직렬화

### 직렬화 가능한 타입

체크포인터를 사용하려면 상태가 직렬화 가능해야 합니다:

```python
# ✅ 직렬화 가능
state = {
    "messages": [HumanMessage(content="안녕")],
    "data": {"key": "value"},
    "count": 42,
}

# ❌ 직렬화 불가
state = {
    "messages": [...],
    "callback": lambda x: x,  # 함수 불가
    "connection": db_conn,     # 연결 객체 불가
}
```

### 커스텀 직렬화

```python
from pydantic import BaseModel

class MyData(BaseModel):
    """Pydantic 모델은 자동 직렬화 지원"""
    value: str
    count: int

class MyState(AgentState):
    data: MyData  # Pydantic 모델 사용
```

---

## 상태 흐름 예시

### 단일 턴 실행

```python
# 1. 초기 상태
initial_state = {
    "messages": [HumanMessage(content="Python 함수를 작성해주세요")]
}

# 2. 에이전트 실행
result = agent.invoke(initial_state)

# 3. 최종 상태
final_state = {
    "messages": [
        HumanMessage(content="Python 함수를 작성해주세요"),
        AIMessage(content="다음은 함수입니다:\n```python\n..."),
    ],
    "todos": [],
    "memory_contents": {...},
    "skills_metadata": [...],
}
```

### 서브에이전트 상태 격리

```python
# 메인 에이전트 상태
main_state = {
    "messages": [HumanMessage, AIMessage, ...],
    "todos": [...],
    "memory_contents": {...},
}

# task() 도구 호출 시 서브에이전트 상태
# 일부 키는 제외됨 (messages, todos, ...)
subagent_state = {
    k: v for k, v in main_state.items()
    if k not in _EXCLUDED_STATE_KEYS
}
# messages는 새로 시작
subagent_state["messages"] = [HumanMessage(content="서브태스크 설명")]

# 서브에이전트 완료 후
# 최종 메시지만 메인에게 반환
```

---

## 컨텍스트 관리

### 토큰 제한

LLM은 컨텍스트 윈도우 제한이 있습니다:
- Claude Sonnet: ~200K 토큰
- GPT-4: ~128K 토큰

### SummarizationMiddleware

긴 대화를 자동으로 요약합니다:

```python
from deepagents.middleware.summarization import SummarizationMiddleware

middleware = SummarizationMiddleware(
    model=model,
    backend=backend,
    trigger=0.8,         # 컨텍스트 80% 사용 시 트리거
    keep=3,              # 최근 3개 메시지 유지
    trim_tokens_to_summarize=None,
)
```

**동작 방식**:
1. 컨텍스트 사용량 모니터링
2. 임계값 초과 시 오래된 메시지 요약
3. 요약을 시스템 메시지로 추가
4. 원본 메시지 삭제

---

## 스토어 (Store)

장기 영속 저장소입니다.

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

agent = create_deep_agent(
    store=store,
)

# 에이전트가 store에 데이터 저장/조회 가능
```

**체크포인터 vs 스토어**:

| 구분 | Checkpointer | Store |
|------|-------------|-------|
| 용도 | 대화 상태 | 장기 데이터 |
| 범위 | 세션 (thread_id) | 전역 |
| 데이터 | 메시지 히스토리 | 임의 키-값 |

---

## 캐시 (Cache)

모델 응답을 캐시합니다.

```python
from langgraph.cache import InMemoryCache

agent = create_deep_agent(
    cache=InMemoryCache(),
)
```

**장점**:
- 동일 입력에 대한 빠른 응답
- API 비용 절감
- 테스트 효율성

---

## 디버깅

### debug 모드

```python
agent = create_deep_agent(debug=True)
```

상태 변화를 상세히 로깅합니다.

### 상태 검사

```python
# 스트리밍으로 중간 상태 확인
async for chunk in agent.astream(
    {"messages": [...]},
    stream_mode="values",
):
    print("Current state:", chunk)
```

---

## 모범 사례

### 1. 상태 최소화

```python
# ✅ 좋은 예: 필요한 데이터만 저장
class MyState(AgentState):
    result_id: str  # ID만 저장

# ❌ 나쁜 예: 큰 데이터 저장
class MyState(AgentState):
    full_result: dict  # 전체 결과 저장
```

### 2. PrivateStateAttr 활용

```python
# 미들웨어 내부 데이터는 비공개로
class MyState(AgentState):
    internal_cache: NotRequired[Annotated[dict, PrivateStateAttr]]
```

### 3. 체크포인터 선택

| 환경 | 권장 체크포인터 |
|------|----------------|
| 개발 | MemorySaver |
| 테스트 | SqliteSaver |
| 프로덕션 | PostgresSaver |

---

## 다음 단계

- [미들웨어 시스템](./middleware-system.md)
- [백엔드 시스템](./backend-system.md)
- [컨텍스트 관리 패턴](../04-patterns/context-management.md)
