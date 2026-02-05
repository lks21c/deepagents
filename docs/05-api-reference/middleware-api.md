# 미들웨어 API 레퍼런스

> Deep Agents 미들웨어 클래스들의 API 레퍼런스입니다.

## AgentMiddleware 베이스 클래스

모든 미들웨어의 기본 인터페이스입니다.

```python
class AgentMiddleware(ABC):
    """에이전트 미들웨어 기본 클래스"""

    state_schema: type[AgentState] | None = None
    """상태 스키마 확장"""

    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> StateUpdate | None:
        """에이전트 실행 전 호출"""
        pass

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """모델 호출을 감싸서 수정"""
        return handler(request)

    def wrap_tool_call(
        self,
        call: ToolCall,
        handler: Callable[[ToolCall], ToolResult],
        runtime: ToolRuntime,
    ) -> ToolResult:
        """도구 호출을 감싸서 수정"""
        return handler(call)
```

---

## TodoListMiddleware

할 일 목록 관리 기능을 제공합니다.

```python
from langchain.agents.middleware import TodoListMiddleware

middleware = TodoListMiddleware()
```

### 제공 도구

- `write_todos`: 할 일 목록 생성/업데이트

### 상태 확장

```python
class TodoState(AgentState):
    todos: NotRequired[Annotated[list[Todo], PrivateStateAttr]]
```

---

## MemoryMiddleware

AGENTS.md 파일에서 메모리를 로드합니다.

```python
from deepagents.middleware.memory import MemoryMiddleware

middleware = MemoryMiddleware(
    backend=backend,
    sources=[
        "~/.deepagents/AGENTS.md",
        "./AGENTS.md",
    ],
)
```

### 파라미터

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `backend` | `BACKEND_TYPES` | 파일 읽기 백엔드 |
| `sources` | `list[str]` | 메모리 파일 경로 목록 |

### 상태 확장

```python
class MemoryState(AgentState):
    memory_contents: NotRequired[Annotated[dict[str, str], PrivateStateAttr]]
```

### 동작

1. `before_agent`: 소스에서 메모리 로드
2. `wrap_model_call`: 시스템 프롬프트에 메모리 주입

---

## SkillsMiddleware

SKILL.md 파일에서 스킬을 로드합니다.

```python
from deepagents.middleware.skills import SkillsMiddleware

middleware = SkillsMiddleware(
    backend=backend,
    sources=[
        "/skills/base/",
        "/skills/user/",
    ],
)
```

### 파라미터

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `backend` | `BACKEND_TYPES` | 파일 읽기 백엔드 |
| `sources` | `list[str]` | 스킬 디렉토리 경로 목록 |

### 상태 확장

```python
class SkillsState(AgentState):
    skills_metadata: NotRequired[Annotated[list[SkillMetadata], PrivateStateAttr]]
```

### 동작

1. `before_agent`: 스킬 메타데이터 로드
2. `wrap_model_call`: 시스템 프롬프트에 스킬 목록 주입

---

## FilesystemMiddleware

파일 시스템 도구를 제공합니다.

```python
from deepagents.middleware.filesystem import FilesystemMiddleware

middleware = FilesystemMiddleware(backend=backend)
```

### 파라미터

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `backend` | `BACKEND_TYPES` | 파일 작업 백엔드 |

### 제공 도구

| 도구 | 설명 |
|------|------|
| `ls` | 디렉토리 목록 |
| `read_file` | 파일 읽기 |
| `write_file` | 파일 생성 |
| `edit_file` | 파일 편집 |
| `glob` | 패턴 매칭 |
| `grep` | 텍스트 검색 |
| `execute` | 명령 실행 (SandboxBackend 필요) |

---

## SubAgentMiddleware

서브에이전트 관리 기능을 제공합니다.

```python
from deepagents.middleware.subagents import SubAgentMiddleware

middleware = SubAgentMiddleware(
    backend=backend,
    subagents=[researcher, coder],
)
```

### 파라미터

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `backend` | `BACKEND_TYPES` | 백엔드 |
| `subagents` | `list[SubAgent \| CompiledSubAgent]` | 서브에이전트 목록 |

### 제공 도구

- `task`: 서브에이전트 호출

---

## SummarizationMiddleware

긴 대화를 자동으로 요약합니다.

```python
from deepagents.middleware.summarization import SummarizationMiddleware

middleware = SummarizationMiddleware(
    model=model,
    backend=backend,
    trigger=0.8,
    keep=3,
    trim_tokens_to_summarize=None,
    truncate_args_settings=None,
)
```

### 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| `model` | `BaseChatModel` | 필수 | 요약에 사용할 모델 |
| `backend` | `BACKEND_TYPES` | 필수 | 백엔드 |
| `trigger` | `float` | 0.8 | 요약 트리거 임계값 (0.0-1.0) |
| `keep` | `int` | 3 | 유지할 최근 메시지 수 |
| `trim_tokens_to_summarize` | `int \| None` | None | 요약할 토큰 수 (자동 계산) |
| `truncate_args_settings` | `dict \| None` | None | 도구 인자 자르기 설정 |

### 동작

컨텍스트 사용량이 `trigger` 임계값을 초과하면:
1. 오래된 메시지 요약
2. 요약을 시스템 메시지로 추가
3. 원본 메시지 삭제
4. 최근 `keep`개 메시지 유지

---

## AnthropicPromptCachingMiddleware

Anthropic 모델의 프롬프트 캐싱을 최적화합니다.

```python
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

middleware = AnthropicPromptCachingMiddleware(
    unsupported_model_behavior="ignore",
)
```

### 파라미터

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `unsupported_model_behavior` | `str` | 비지원 모델 동작 (`"ignore"` \| `"error"`) |

---

## PatchToolCallsMiddleware

도구 호출의 JSON 스키마 호환성을 처리합니다.

```python
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware

middleware = PatchToolCallsMiddleware()
```

---

## HumanInTheLoopMiddleware

도구 실행 전 사용자 승인을 요청합니다.

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        "edit_file": True,
        "execute": InterruptOnConfig(
            condition=lambda args: "rm" in args.get("command", ""),
        ),
    },
)
```

### 파라미터

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `interrupt_on` | `dict[str, bool \| InterruptOnConfig]` | 도구별 인터럽트 설정 |

### InterruptOnConfig

```python
class InterruptOnConfig(TypedDict):
    condition: Callable[[dict], bool] | None
    """인터럽트 조건 함수"""

    message: str | None
    """사용자에게 표시할 메시지"""
```

---

## 미들웨어 실행 순서

미들웨어는 등록 순서대로 체인 형태로 실행됩니다:

```
요청 → MW1 → MW2 → MW3 → 모델/도구 → MW3 → MW2 → MW1 → 응답
```

**기본 스택 순서**:
1. TodoListMiddleware
2. MemoryMiddleware (선택)
3. SkillsMiddleware (선택)
4. FilesystemMiddleware
5. SubAgentMiddleware
6. SummarizationMiddleware
7. AnthropicPromptCachingMiddleware
8. PatchToolCallsMiddleware
9. 사용자 미들웨어
10. HumanInTheLoopMiddleware (선택)

---

## 커스텀 미들웨어 작성

```python
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse

class MyMiddleware(AgentMiddleware):
    """커스텀 미들웨어 예시"""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # 요청 수정
        modified_request = request.override(
            system_message=request.system_message + "\n[커스텀 지시]"
        )

        # 핸들러 호출
        response = handler(modified_request)

        # 응답 처리
        return response
```

---

## 관련 문서

- [미들웨어 시스템](../01-architecture/middleware-system.md)
- [커스텀 미들웨어 작성](../06-advanced/custom-middleware.md)
- [create_deep_agent API](./create-deep-agent-api.md)
