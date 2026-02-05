# create_deep_agent API 레퍼런스

> `create_deep_agent()` 함수의 전체 파라미터 레퍼런스입니다.

## 함수 시그니처

**소스 위치**: `libs/deepagents/deepagents/graph.py:96-376`

```python
def create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph:
```

---

## 파라미터

### model

```python
model: str | BaseChatModel | None = None
```

사용할 LLM 모델입니다.

| 타입 | 설명 |
|------|------|
| `None` | 기본값: `claude-sonnet-4-5-20250929` |
| `str` | 모델 문자열 (예: `"openai:gpt-4"`, `"anthropic:claude-3-opus"`) |
| `BaseChatModel` | LangChain 모델 인스턴스 |

**예시**:

```python
# 기본 (Claude Sonnet 4.5)
agent = create_deep_agent()

# 문자열로 지정
agent = create_deep_agent(model="openai:gpt-4o")

# 직접 인스턴스
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model_name="claude-3-opus-20240229")
agent = create_deep_agent(model=model)
```

---

### tools

```python
tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None
```

에이전트가 사용할 커스텀 도구 목록입니다.

| 타입 | 설명 |
|------|------|
| `BaseTool` | LangChain 도구 인스턴스 |
| `Callable` | `@tool` 데코레이터가 적용된 함수 |
| `dict` | 도구 정의 딕셔너리 |

**기본 제공 도구**:
- `write_todos`: 할 일 목록 관리
- `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`: 파일 작업
- `execute`: 셸 명령 실행 (SandboxBackendProtocol 필요)
- `task`: 서브에이전트 호출

**예시**:

```python
from langchain_core.tools import tool

@tool
def my_tool(arg: str) -> str:
    """커스텀 도구"""
    return f"결과: {arg}"

agent = create_deep_agent(tools=[my_tool])
```

---

### system_prompt

```python
system_prompt: str | SystemMessage | None = None
```

에이전트의 시스템 프롬프트입니다.

| 타입 | 설명 |
|------|------|
| `None` | 기본 프롬프트만 사용 |
| `str` | 기본 프롬프트 앞에 추가 |
| `SystemMessage` | content_blocks에 기본 프롬프트 추가 |

**기본 프롬프트**:
```
"In order to complete the objective that the user asks of you, you have access to a number of standard tools."
```

**예시**:

```python
# 문자열
agent = create_deep_agent(
    system_prompt="당신은 Python 전문가입니다."
)

# SystemMessage
from langchain_core.messages import SystemMessage
agent = create_deep_agent(
    system_prompt=SystemMessage(content=[
        {"type": "text", "text": "당신은 Python 전문가입니다."},
        {"type": "image", "source": {...}},
    ])
)
```

---

### middleware

```python
middleware: Sequence[AgentMiddleware] = ()
```

기본 스택 이후에 적용할 추가 미들웨어입니다.

**기본 미들웨어 스택**:
1. `TodoListMiddleware`
2. `MemoryMiddleware` (memory 지정 시)
3. `SkillsMiddleware` (skills 지정 시)
4. `FilesystemMiddleware`
5. `SubAgentMiddleware`
6. `SummarizationMiddleware`
7. `AnthropicPromptCachingMiddleware`
8. `PatchToolCallsMiddleware`
9. `HumanInTheLoopMiddleware` (interrupt_on 지정 시)

**예시**:

```python
from langchain.agents.middleware import MyCustomMiddleware

agent = create_deep_agent(
    middleware=[MyCustomMiddleware()],
)
```

---

### subagents

```python
subagents: list[SubAgent | CompiledSubAgent] | None = None
```

사용할 서브에이전트 목록입니다.

**SubAgent 필수 필드**:
- `name`: 고유 식별자
- `description`: 설명 (메인 에이전트가 위임 결정에 사용)
- `system_prompt`: 시스템 프롬프트

**SubAgent 선택 필드**:
- `tools`: 사용할 도구 (기본: 메인 에이전트 도구 상속)
- `model`: 모델 (기본: 메인 에이전트 모델 상속)
- `middleware`: 추가 미들웨어
- `interrupt_on`: HITL 설정
- `skills`: 스킬 경로

**예시**:

```python
researcher = {
    "name": "researcher",
    "description": "웹 리서치를 수행하는 에이전트",
    "system_prompt": "당신은 리서치 전문가입니다.",
    "tools": [web_search],
}

agent = create_deep_agent(subagents=[researcher])
```

---

### skills

```python
skills: list[str] | None = None
```

스킬 소스 경로 목록입니다.

**경로 규칙**:
- POSIX 스타일 슬래시 사용
- 같은 이름의 스킬은 마지막 것이 우선

**예시**:

```python
agent = create_deep_agent(
    skills=[
        "/skills/base/",      # 기본 스킬
        "/skills/user/",      # 사용자 스킬 (우선)
    ],
)
```

---

### memory

```python
memory: list[str] | None = None
```

메모리 파일(AGENTS.md) 경로 목록입니다.

**예시**:

```python
agent = create_deep_agent(
    memory=[
        "~/.deepagents/AGENTS.md",
        "./project/AGENTS.md",
    ],
)
```

---

### response_format

```python
response_format: ResponseFormat | None = None
```

구조화된 출력 형식입니다.

**예시**:

```python
from pydantic import BaseModel

class Response(BaseModel):
    answer: str
    confidence: float

agent = create_deep_agent(
    response_format=Response,
)
```

---

### context_schema

```python
context_schema: type[Any] | None = None
```

에이전트 컨텍스트 스키마 타입입니다.

---

### checkpointer

```python
checkpointer: Checkpointer | None = None
```

실행 간 상태 유지를 위한 체크포인터입니다.

**옵션**:
- `MemorySaver`: 인메모리 (개발용)
- `SqliteSaver`: SQLite (테스트용)
- `PostgresSaver`: PostgreSQL (프로덕션용)

**예시**:

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_deep_agent(
    checkpointer=MemorySaver(),
)

# 세션 유지
result = agent.invoke(
    {"messages": [...]},
    config={"configurable": {"thread_id": "session-1"}}
)
```

---

### store

```python
store: BaseStore | None = None
```

영구 저장소입니다.

**예시**:

```python
from langgraph.store.memory import InMemoryStore

agent = create_deep_agent(
    store=InMemoryStore(),
)
```

---

### backend

```python
backend: BackendProtocol | BackendFactory | None = None
```

파일 저장 및 실행 백엔드입니다.

| 타입 | 설명 |
|------|------|
| `None` | 기본값: `StateBackend` |
| `BackendProtocol` | 직접 인스턴스 |
| `BackendFactory` | 팩토리 함수 |

**예시**:

```python
from deepagents.backends import FilesystemBackend

# 직접 인스턴스
agent = create_deep_agent(
    backend=FilesystemBackend(root_dir="./workspace"),
)

# 팩토리 함수
from deepagents.backends import StateBackend
agent = create_deep_agent(
    backend=lambda rt: StateBackend(rt),
)
```

---

### interrupt_on

```python
interrupt_on: dict[str, bool | InterruptOnConfig] | None = None
```

도구별 인터럽트 설정입니다.

**예시**:

```python
from langchain.agents.middleware import InterruptOnConfig

agent = create_deep_agent(
    checkpointer=MemorySaver(),  # 필수
    interrupt_on={
        "edit_file": True,
        "execute": InterruptOnConfig(
            condition=lambda args: "rm" in args.get("command", ""),
            message="위험한 명령입니다. 실행하시겠습니까?",
        ),
    },
)
```

---

### debug

```python
debug: bool = False
```

디버그 모드 활성화 여부입니다.

---

### name

```python
name: str | None = None
```

에이전트 이름입니다.

---

### cache

```python
cache: BaseCache | None = None
```

모델 응답 캐시입니다.

**예시**:

```python
from langgraph.cache import InMemoryCache

agent = create_deep_agent(
    cache=InMemoryCache(),
)
```

---

## 반환값

```python
CompiledStateGraph
```

구성된 Deep Agent입니다. 재귀 제한은 1000으로 설정됩니다.

---

## 사용 예시

### 기본 사용

```python
from deepagents import create_deep_agent

agent = create_deep_agent()
result = agent.invoke({
    "messages": [{"role": "user", "content": "안녕하세요!"}]
})
```

### 완전한 설정

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[my_tool],
    system_prompt="당신은 Python 전문가입니다.",
    subagents=[researcher],
    skills=["./skills/"],
    memory=["./AGENTS.md"],
    backend=FilesystemBackend(root_dir="./workspace"),
    checkpointer=MemorySaver(),
    interrupt_on={"edit_file": True},
    debug=True,
)
```

---

## 관련 문서

- [서브에이전트 API](./subagent-api.md)
- [미들웨어 API](./middleware-api.md)
- [백엔드 API](./backend-api.md)
