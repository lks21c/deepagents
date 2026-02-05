# 서브에이전트 API 레퍼런스

> SubAgent 및 CompiledSubAgent의 API 레퍼런스입니다.

## SubAgent TypedDict

**소스 위치**: `libs/deepagents/deepagents/middleware/subagents.py:22-78`

서브에이전트 사양을 정의하는 TypedDict입니다.

```python
class SubAgent(TypedDict):
    """서브에이전트 사양 정의"""

    # 필수 필드
    name: str
    description: str
    system_prompt: str

    # 선택 필드
    tools: NotRequired[Sequence[BaseTool | Callable | dict[str, Any]]]
    model: NotRequired[str | BaseChatModel]
    middleware: NotRequired[list[AgentMiddleware]]
    interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]
    skills: NotRequired[list[str]]
```

---

## 필수 필드

### name

```python
name: str
```

서브에이전트의 고유 식별자입니다.

**요구사항**:
- 고유해야 함
- `task()` 호출 시 `subagent_type`으로 사용

**예시**:
```python
{"name": "researcher"}
{"name": "code-reviewer"}
```

---

### description

```python
description: str
```

서브에이전트의 기능 설명입니다.

**용도**:
- 메인 에이전트가 위임 결정에 사용
- 어떤 작업에 적합한지 명시

**예시**:
```python
{"description": "웹 리서치를 수행하는 전문 에이전트. 한 번에 하나의 주제만 연구."}
```

---

### system_prompt

```python
system_prompt: str
```

서브에이전트의 시스템 프롬프트입니다.

**내용**:
- 역할 정의
- 작업 방식
- 출력 형식

**예시**:
```python
{
    "system_prompt": """당신은 철저한 리서치를 수행하는 연구원입니다.

## 작업 방식
1. 검색 전략 수립
2. 다양한 쿼리로 정보 수집
3. 수집한 정보 검증 및 정리
4. 구조화된 결과 반환

## 출력 형식
- 핵심 발견사항 요약
- 상세 내용 (출처 포함)
"""
}
```

---

## 선택 필드

### tools

```python
tools: NotRequired[Sequence[BaseTool | Callable | dict[str, Any]]]
```

서브에이전트가 사용할 도구 목록입니다.

**기본값**: 메인 에이전트의 도구 상속

**예시**:
```python
@tool
def web_search(query: str) -> dict:
    """웹 검색"""
    pass

{
    "name": "researcher",
    "tools": [web_search],  # 전용 도구
}
```

---

### model

```python
model: NotRequired[str | BaseChatModel]
```

사용할 LLM 모델입니다.

**기본값**: 메인 에이전트의 모델 상속

**예시**:
```python
# 문자열
{"model": "openai:gpt-4o"}

# 인스턴스
from langchain_openai import ChatOpenAI
{"model": ChatOpenAI(model="gpt-4o")}
```

---

### middleware

```python
middleware: NotRequired[list[AgentMiddleware]]
```

추가 미들웨어 목록입니다.

**기본값**: 기본 미들웨어 스택에 추가됨

**예시**:
```python
from langchain.agents.middleware import MyCustomMiddleware

{
    "name": "custom-agent",
    "middleware": [MyCustomMiddleware()],
}
```

---

### interrupt_on

```python
interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]
```

도구별 인터럽트 설정입니다.

**예시**:
```python
{
    "name": "code-modifier",
    "interrupt_on": {
        "edit_file": True,
    },
}
```

---

### skills

```python
skills: NotRequired[list[str]]
```

스킬 소스 경로 목록입니다.

**예시**:
```python
{
    "name": "content-writer",
    "skills": ["./skills/writing/"],
}
```

---

## CompiledSubAgent TypedDict

**소스 위치**: `libs/deepagents/deepagents/middleware/subagents.py:81-110`

사전 컴파일된 LangGraph 에이전트를 서브에이전트로 사용합니다.

```python
class CompiledSubAgent(TypedDict):
    """사전 컴파일된 에이전트 사양"""

    name: str
    description: str
    runnable: Runnable
```

### 필드

| 필드 | 타입 | 설명 |
|------|------|------|
| `name` | `str` | 고유 식별자 |
| `description` | `str` | 기능 설명 |
| `runnable` | `Runnable` | 컴파일된 에이전트 |

**요구사항**:
- `runnable`은 상태에 `messages` 키를 가져야 함

**예시**:
```python
from langchain.agents import create_agent

# 커스텀 에이전트 컴파일
custom_agent = create_agent(
    model,
    tools=[...],
    system_prompt="...",
)

compiled_subagent: CompiledSubAgent = {
    "name": "custom-agent",
    "description": "커스텀 로직을 가진 에이전트",
    "runnable": custom_agent,
}

agent = create_deep_agent(
    subagents=[compiled_subagent],
)
```

---

## GENERAL_PURPOSE_SUBAGENT

**소스 위치**: `libs/deepagents/deepagents/middleware/subagents.py:268-275`

기본 제공되는 범용 서브에이전트입니다.

```python
GENERAL_PURPOSE_SUBAGENT: SubAgent = {
    "name": "general-purpose",
    "description": "General-purpose agent for researching complex questions, "
                   "searching for files and content, and executing multi-step tasks. "
                   "This agent has access to all tools as the main agent.",
    "system_prompt": DEFAULT_SUBAGENT_PROMPT,
}
```

**특징**:
- 메인 에이전트와 동일한 도구 사용
- 범용 작업 처리
- 자동으로 포함됨 (별도 등록 불필요)

---

## task() 도구

서브에이전트를 호출하는 도구입니다.

```python
task(
    description: str,
    subagent_type: str,
) -> str
```

### 파라미터

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `description` | `str` | 작업 설명 |
| `subagent_type` | `str` | 서브에이전트 이름 |

### 사용 예시 (LLM 관점)

```python
# 메인 에이전트가 생성하는 도구 호출
task(
    description="Python으로 피보나치 함수를 구현해주세요",
    subagent_type="coder"
)
```

---

## 서브에이전트 생명주기

### 1. Spawn (생성)

```python
# task() 호출 시 서브에이전트 생성
task(description="...", subagent_type="researcher")
```

### 2. Run (실행)

서브에이전트는 독립적인 컨텍스트에서 실행됩니다.

**제외되는 상태 키**:
```python
_EXCLUDED_STATE_KEYS = {
    "messages",           # 새 히스토리 시작
    "todos",              # 자체 할 일 관리
    "structured_response",
    "skills_metadata",    # 자체 스킬 로드
    "memory_contents",    # 자체 메모리 로드
}
```

### 3. Return (반환)

최종 메시지만 메인 에이전트에게 반환됩니다.

```python
# 내부 동작
message_text = result["messages"][-1].text.rstrip()
return Command(update={
    "messages": [ToolMessage(message_text, tool_call_id=tool_call_id)],
})
```

---

## 서브에이전트 정의 예시

### 리서치 에이전트

```python
research_agent: SubAgent = {
    "name": "researcher",
    "description": "웹 리서치를 수행하는 전문 에이전트. 한 번에 하나의 주제만 연구.",
    "system_prompt": """당신은 철저한 리서치를 수행하는 연구원입니다.

## 작업 방식
1. think 도구로 검색 전략 수립
2. web_search로 정보 수집
3. 수집한 정보 검증 및 정리
4. 구조화된 결과 반환

## 출력 형식
- 핵심 발견사항 요약
- 상세 내용 (출처 포함)
""",
    "tools": [web_search, think],
}
```

### 코드 리뷰어

```python
code_reviewer: SubAgent = {
    "name": "code-reviewer",
    "description": "코드 품질, 보안, 성능을 검토하는 에이전트",
    "system_prompt": """당신은 시니어 개발자로서 코드를 검토합니다.

## 검토 항목
1. 코드 품질: 가독성, 유지보수성
2. 버그 가능성: 엣지 케이스, 에러 처리
3. 보안: 인젝션, 인증/인가
4. 성능: 알고리즘 복잡도

## 출력 형식
각 이슈에 대해:
- 위치 (파일:라인)
- 심각도 (Critical/High/Medium/Low)
- 문제 설명
- 개선 제안
""",
    "model": "openai:gpt-4o",
}
```

---

## 모범 사례

### 1. 명확한 description

```python
# ✅ 좋은 예
{"description": "웹 리서치를 수행하는 에이전트. 한 번에 하나의 주제만 연구."}

# ❌ 나쁜 예
{"description": "리서치 에이전트"}
```

### 2. 구체적인 system_prompt

```python
# 역할, 작업 방식, 출력 형식을 모두 명시
{
    "system_prompt": """
## 역할
...

## 작업 방식
1. ...
2. ...

## 출력 형식
- ...
"""
}
```

### 3. 적절한 도구 제공

```python
# 필요한 도구만 제공
{"tools": [web_search]}  # 리서치에 필요한 도구만

# 메인 에이전트 도구 상속
{"tools": None}  # 또는 생략
```

---

## 관련 문서

- [서브에이전트 오케스트레이션](../02-core-concepts/subagent-orchestration.md)
- [병렬 서브에이전트 패턴](../04-patterns/parallel-subagents.md)
- [create_deep_agent API](./create-deep-agent-api.md)
