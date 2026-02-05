# create_deep_agent() 심층 분석

> Deep Agents의 핵심 팩토리 함수인 `create_deep_agent()`를 라인별로 상세하게 분석합니다.

## 개요

`create_deep_agent()`는 Deep Agents 프레임워크의 진입점입니다. 이 함수는 LangChain의 `create_agent()`를 래핑하여 계획 수립, 파일 시스템 접근, 서브에이전트 관리 등의 고급 기능을 기본으로 제공합니다.

**소스 위치**: `libs/deepagents/deepagents/graph.py`

## 함수 시그니처

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

## 파라미터 상세 설명

### 필수 파라미터 (선택적 기본값 제공)

#### `model`
```python
model: str | BaseChatModel | None = None
```

**설명**: 에이전트가 사용할 LLM 모델입니다.

**타입 옵션**:
- `None`: 기본값으로 Claude Sonnet 4.5 사용
- `str`: `"provider:model"` 형식 (예: `"openai:gpt-4o"`, `"anthropic:claude-sonnet-4-5-20250929"`)
- `BaseChatModel`: LangChain 모델 인스턴스

**기본값 처리 코드** (graph.py:190-193):
```python
if model is None:
    model = get_default_model()  # Claude Sonnet 4.5 반환
elif isinstance(model, str):
    model = init_chat_model(model)  # 문자열에서 모델 인스턴스 생성
```

**설계 이유**:
- **유연성**: 문자열 기반 지정으로 빠른 모델 전환 가능
- **합리적 기본값**: Claude Sonnet 4.5는 도구 호출과 복잡한 추론에 뛰어난 성능 제공
- **호환성**: 어떤 LangChain 호환 모델이든 `BaseChatModel`로 직접 전달 가능

**사용 예시**:
```python
# 기본값 (Claude Sonnet 4.5)
agent = create_deep_agent()

# 문자열로 모델 지정
agent = create_deep_agent(model="openai:gpt-4o")

# 직접 모델 인스턴스 전달
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
agent = create_deep_agent(model=model)
```

---

#### `tools`
```python
tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None
```

**설명**: 에이전트가 사용할 커스텀 도구 목록입니다.

**타입 옵션**:
- `BaseTool`: LangChain 도구 인스턴스
- `Callable`: `@tool` 데코레이터가 적용된 함수
- `dict`: 도구 사양 딕셔너리

**중요**: 기본 제공 도구(`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `task`, `write_todos`)는 자동으로 포함됩니다. 이 파라미터는 **추가** 도구를 지정합니다.

**서브에이전트로의 상속** (graph.py:223-224, 263):
```python
general_purpose_spec: SubAgent = {
    ...
    "tools": tools or [],  # 커스텀 도구를 general-purpose 서브에이전트에도 전달
}

# 사용자 정의 서브에이전트도 미지정시 상속
processed_spec: SubAgent = {
    ...
    "tools": spec.get("tools", tools or []),
}
```

**설계 이유**:
- **도구 상속**: 기본적으로 모든 서브에이전트가 메인 에이전트의 도구를 상속
- **유연한 오버라이드**: 각 서브에이전트가 필요시 자체 도구 세트 지정 가능

**사용 예시**:
```python
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """웹에서 정보를 검색합니다."""
    return search_api(query)

@tool
def calculator(expression: str) -> float:
    """수학 표현식을 계산합니다."""
    return eval(expression)

agent = create_deep_agent(tools=[web_search, calculator])
```

---

### 시스템 프롬프트

#### `system_prompt`
```python
system_prompt: str | SystemMessage | None = None
```

**설명**: 에이전트의 동작을 정의하는 시스템 프롬프트입니다.

**프롬프트 결합 로직** (graph.py:303-315):
```python
BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."

if system_prompt is None:
    final_system_prompt: str | SystemMessage = BASE_AGENT_PROMPT
elif isinstance(system_prompt, SystemMessage):
    # SystemMessage: content_blocks에 BASE_AGENT_PROMPT 추가
    new_content = [
        *system_prompt.content_blocks,
        {"type": "text", "text": f"\n\n{BASE_AGENT_PROMPT}"},
    ]
    final_system_prompt = SystemMessage(content=new_content)
else:
    # String: 단순 연결
    final_system_prompt = system_prompt + "\n\n" + BASE_AGENT_PROMPT
```

**설계 이유**:
- **기본 지침 보장**: 항상 기본 도구 사용 지침이 포함됨
- **사용자 프롬프트 우선**: 사용자 프롬프트가 먼저 오고 기본 프롬프트가 뒤에 추가됨
- **멀티모달 지원**: `SystemMessage`를 통해 이미지 등 멀티모달 컨텐츠 포함 가능

**사용 예시**:
```python
# 문자열 프롬프트
agent = create_deep_agent(
    system_prompt="당신은 Python 전문가입니다. 항상 PEP 8을 준수하세요."
)

# SystemMessage (멀티모달)
from langchain_core.messages import SystemMessage
agent = create_deep_agent(
    system_prompt=SystemMessage(content=[
        {"type": "text", "text": "당신은 UI 디자이너입니다."},
        {"type": "image_url", "image_url": {"url": "brand_guidelines.png"}},
    ])
)
```

---

### 미들웨어와 확장

#### `middleware`
```python
middleware: Sequence[AgentMiddleware] = ()
```

**설명**: 기본 미들웨어 스택 **이후에** 추가될 커스텀 미들웨어입니다.

**적용 순서** (graph.py:298-299):
```python
if middleware:
    deepagent_middleware.extend(middleware)  # 기본 스택 뒤에 추가
```

**기본 미들웨어 순서** (graph.py:272-296):
1. `TodoListMiddleware`
2. `MemoryMiddleware` (if memory)
3. `SkillsMiddleware` (if skills)
4. `FilesystemMiddleware`
5. `SubAgentMiddleware`
6. `SummarizationMiddleware`
7. `AnthropicPromptCachingMiddleware`
8. `PatchToolCallsMiddleware`
9. **[사용자 정의 미들웨어]** ← 여기에 추가됨
10. `HumanInTheLoopMiddleware` (if interrupt_on)

**설계 이유**:
- **순서 보장**: HITL은 항상 마지막에 와야 하므로 사용자 미들웨어는 그 전에 삽입
- **기본 스택 보존**: 핵심 기능이 먼저 설정된 후 사용자 확장이 적용됨

**사용 예시**:
```python
class LoggingMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        print(f"LLM 호출: {len(request.messages)} 메시지")
        response = handler(request)
        print(f"응답: {response.response.content[:100]}...")
        return response

agent = create_deep_agent(middleware=[LoggingMiddleware()])
```

---

#### `subagents`
```python
subagents: list[SubAgent | CompiledSubAgent] | None = None
```

**설명**: 작업을 위임받을 서브에이전트 목록입니다.

**SubAgent TypedDict 구조** (subagents.py:22-78):
```python
class SubAgent(TypedDict):
    name: str          # 고유 식별자 (필수)
    description: str   # 메인 에이전트가 위임 결정시 사용 (필수)
    system_prompt: str # 서브에이전트 시스템 프롬프트 (필수)

    # 선택적 필드
    tools: NotRequired[Sequence[BaseTool | Callable | dict]]
    model: NotRequired[str | BaseChatModel]
    middleware: NotRequired[list[AgentMiddleware]]
    interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]
    skills: NotRequired[list[str]]
```

**서브에이전트 처리 로직** (graph.py:228-266):
```python
processed_subagents: list[SubAgent | CompiledSubAgent] = []
for spec in subagents or []:
    if "runnable" in spec:
        # CompiledSubAgent는 그대로 사용
        processed_subagents.append(spec)
    else:
        # SubAgent: 기본값 채우기와 미들웨어 스택 빌드
        subagent_model = spec.get("model", model)  # 미지정시 메인 모델 상속
        if isinstance(subagent_model, str):
            subagent_model = init_chat_model(subagent_model)

        # 기본 미들웨어 스택 구성
        subagent_middleware: list[AgentMiddleware] = [
            TodoListMiddleware(),
            FilesystemMiddleware(backend=backend),
            SummarizationMiddleware(model=subagent_model, backend=backend, ...),
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
            PatchToolCallsMiddleware(),
        ]
        # 스킬이 있으면 추가
        subagent_skills = spec.get("skills")
        if subagent_skills:
            subagent_middleware.append(SkillsMiddleware(backend=backend, sources=subagent_skills))
        # 사용자 정의 미들웨어 추가
        subagent_middleware.extend(spec.get("middleware", []))

        processed_spec: SubAgent = {
            **spec,
            "model": subagent_model,
            "tools": spec.get("tools", tools or []),  # 미지정시 메인 도구 상속
            "middleware": subagent_middleware,
        }
        processed_subagents.append(processed_spec)
```

**general-purpose 서브에이전트** (graph.py:220-225, 268-269):
```python
# 기본으로 포함되는 범용 서브에이전트
GENERAL_PURPOSE_SUBAGENT: SubAgent = {
    "name": "general-purpose",
    "description": "General-purpose agent for researching complex questions...",
    "system_prompt": DEFAULT_SUBAGENT_PROMPT,
}

# 모든 서브에이전트 조합
all_subagents: list[SubAgent | CompiledSubAgent] = [general_purpose_spec, *processed_subagents]
```

**설계 이유**:
- **기본값 상속**: 모델, 도구, 미들웨어가 미지정시 메인 에이전트로부터 상속
- **완전한 미들웨어 스택**: 각 서브에이전트도 자체 미들웨어 스택을 가짐
- **general-purpose 보장**: 항상 기본 범용 서브에이전트가 포함됨

**사용 예시**:
```python
research_agent = {
    "name": "researcher",
    "description": "웹 리서치를 수행하는 전문 에이전트",
    "system_prompt": """당신은 철저한 리서치를 수행하는 연구원입니다.
    검색 도구를 사용하여 정보를 수집하고 정리하세요.""",
    "tools": [web_search_tool],
}

code_reviewer = {
    "name": "code-reviewer",
    "description": "코드 품질과 보안을 검토하는 에이전트",
    "system_prompt": "당신은 시니어 개발자입니다. 코드를 검토하고 개선점을 제안하세요.",
    "model": "openai:gpt-4o",  # 다른 모델 사용
}

agent = create_deep_agent(subagents=[research_agent, code_reviewer])
```

---

### 파일 기반 설정

#### `skills`
```python
skills: list[str] | None = None
```

**설명**: SKILL.md 파일들이 위치한 디렉토리 경로 목록입니다.

**스킬 로딩 방식**:
- 각 경로에서 `SKILL.md` 파일을 가진 하위 디렉토리를 검색
- YAML frontmatter에서 메타데이터(name, description) 파싱
- 시스템 프롬프트에 스킬 목록 주입 (점진적 공개 패턴)

**우선순위** (skills.py:617-621):
```python
# 나중 소스가 이전 소스를 오버라이드 (last one wins)
for source_path in self.sources:
    source_skills = _list_skills(backend, source_path)
    for skill in source_skills:
        all_skills[skill["name"]] = skill  # 같은 이름이면 덮어씀
```

**설계 이유**:
- **레이어링**: `base → user → project` 순으로 스킬 오버라이드 가능
- **점진적 공개**: 전체 스킬 내용 대신 메타데이터만 노출하여 컨텍스트 절약

**사용 예시**:
```python
# 디렉토리 구조
# ./skills/
# ├── web-research/
# │   └── SKILL.md
# └── code-review/
#     └── SKILL.md

agent = create_deep_agent(
    skills=["./skills/"],
    backend=FilesystemBackend(root_dir="."),
)
```

---

#### `memory`
```python
memory: list[str] | None = None
```

**설명**: AGENTS.md 파일 경로 목록입니다.

**메모리 로딩 및 주입** (memory.py:303-330):
```python
def before_agent(self, state, runtime, config) -> MemoryStateUpdate | None:
    if "memory_contents" in state:
        return None  # 이미 로드됨

    backend = self._get_backend(state, runtime, config)
    contents: dict[str, str] = {}

    for path in self.sources:
        content = self._load_memory_from_backend_sync(backend, path)
        if content:
            contents[path] = content

    return MemoryStateUpdate(memory_contents=contents)
```

**시스템 프롬프트 주입** (memory.py:93-152):
```python
MEMORY_SYSTEM_PROMPT = """<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
    The above <agent_memory> was loaded in from files in your filesystem...
    **When to update memories:**
    - When the user explicitly asks you to remember something
    - When the user describes your role or how you should behave
    ...
</memory_guidelines>
"""
```

**설계 이유**:
- **영속 컨텍스트**: 에이전트가 세션 간에 학습한 내용을 기억
- **자동 업데이트 지침**: 에이전트에게 언제 메모리를 업데이트해야 하는지 가이드

**사용 예시**:
```python
# AGENTS.md 예시 내용:
# # 에이전트 지침
# ## 코딩 스타일
# - PEP 8 준수
# - Type hints 필수

agent = create_deep_agent(
    memory=["./AGENTS.md"],
    backend=FilesystemBackend(root_dir="."),
)
```

---

### 상태와 저장소

#### `backend`
```python
backend: BackendProtocol | BackendFactory | None = None
```

**설명**: 파일 저장소 및 명령 실행 백엔드입니다.

**기본값 처리** (graph.py:198):
```python
backend = backend if backend is not None else (lambda rt: StateBackend(rt))
```

**백엔드 팩토리 패턴**:
```python
# StateBackend는 런타임에 생성되어야 함 (상태 접근 필요)
backend = lambda rt: StateBackend(rt)

# FilesystemBackend는 직접 인스턴스 가능
backend = FilesystemBackend(root_dir="/workspace")
```

**설계 이유**:
- **팩토리 지원**: `StateBackend`처럼 런타임 정보가 필요한 백엔드를 위한 지연 생성
- **추상화**: 어떤 저장소든 `BackendProtocol`만 구현하면 사용 가능

**사용 예시**:
```python
from deepagents.backends import FilesystemBackend, StateBackend

# 로컬 파일 시스템
agent = create_deep_agent(
    backend=FilesystemBackend(root_dir="/workspace"),
)

# 하이브리드 (CompositeBackend)
from deepagents.backends import CompositeBackend
agent = create_deep_agent(
    backend=CompositeBackend(
        default=lambda rt: StateBackend(rt),
        routes={"/memories/": FilesystemBackend(root_dir="./memories")},
    ),
)
```

---

#### `checkpointer`
```python
checkpointer: Checkpointer | None = None
```

**설명**: 에이전트 상태를 영속화하기 위한 LangGraph Checkpointer입니다.

**사용 사례**:
- 대화 히스토리 영속화
- Human-in-the-loop 워크플로우 지원
- 에이전트 상태 복원

**사용 예시**:
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
agent = create_deep_agent(checkpointer=checkpointer)

# thread_id로 대화 지속
result1 = agent.invoke(
    {"messages": [("user", "안녕")]},
    config={"configurable": {"thread_id": "session-1"}}
)

result2 = agent.invoke(
    {"messages": [("user", "방금 뭐라고 했어?")]},
    config={"configurable": {"thread_id": "session-1"}}  # 같은 세션
)
```

---

#### `store`
```python
store: BaseStore | None = None
```

**설명**: 영속 키-값 저장소입니다. `StoreBackend`와 함께 사용합니다.

---

### 제어 옵션

#### `interrupt_on`
```python
interrupt_on: dict[str, bool | InterruptOnConfig] | None = None
```

**설명**: 특정 도구 호출 전에 에이전트 실행을 일시 중지합니다.

**HITL 미들웨어 추가** (graph.py:300-301):
```python
if interrupt_on is not None:
    deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))
```

**설계 이유**:
- **안전성**: 위험한 작업(파일 수정, 외부 API 호출) 전에 사람 승인 요청
- **디버깅**: 에이전트 동작을 단계별로 검사

**사용 예시**:
```python
agent = create_deep_agent(
    interrupt_on={
        "edit_file": True,      # 모든 파일 편집 전 중단
        "execute": True,        # 모든 명령 실행 전 중단
        "task": {               # 서브에이전트 호출 시 조건부 중단
            "requires_approval": lambda args: "delete" in args.get("description", ""),
        },
    },
    checkpointer=MemorySaver(),  # HITL에 필수
)
```

---

## 전체 처리 흐름 요약

```python
def create_deep_agent(...) -> CompiledStateGraph:
    # 1. 모델 해석
    if model is None:
        model = get_default_model()  # Claude Sonnet 4.5
    elif isinstance(model, str):
        model = init_chat_model(model)

    # 2. 요약 설정 계산
    summarization_defaults = _compute_summarization_defaults(model)

    # 3. 백엔드 설정
    backend = backend if backend is not None else (lambda rt: StateBackend(rt))

    # 4. General-purpose 서브에이전트 구성
    gp_middleware = [TodoListMiddleware(), FilesystemMiddleware(...), ...]
    general_purpose_spec = {..., "model": model, "tools": tools or [], ...}

    # 5. 사용자 서브에이전트 처리 (기본값 채우기)
    processed_subagents = []
    for spec in subagents or []:
        # 모델, 도구, 미들웨어 기본값 처리
        processed_subagents.append(processed_spec)

    # 6. 전체 서브에이전트 조합
    all_subagents = [general_purpose_spec, *processed_subagents]

    # 7. 메인 에이전트 미들웨어 스택 구성
    deepagent_middleware = [
        TodoListMiddleware(),
        # MemoryMiddleware (if memory)
        # SkillsMiddleware (if skills)
        FilesystemMiddleware(backend=backend),
        SubAgentMiddleware(backend=backend, subagents=all_subagents),
        SummarizationMiddleware(...),
        AnthropicPromptCachingMiddleware(...),
        PatchToolCallsMiddleware(),
        # [사용자 미들웨어]
        # HumanInTheLoopMiddleware (if interrupt_on)
    ]

    # 8. 시스템 프롬프트 결합
    final_system_prompt = system_prompt + "\n\n" + BASE_AGENT_PROMPT

    # 9. LangChain create_agent 호출
    return create_agent(
        model,
        system_prompt=final_system_prompt,
        tools=tools,
        middleware=deepagent_middleware,
        ...
    ).with_config({"recursion_limit": 1000})
```

---

## 다음 단계

- [서브에이전트 오케스트레이션](./subagent-orchestration.md): 서브에이전트 시스템 상세
- [스킬 시스템](./skills-system.md): SKILL.md 포맷과 활용
- [메모리 시스템](./memory-system.md): AGENTS.md 영속 컨텍스트
