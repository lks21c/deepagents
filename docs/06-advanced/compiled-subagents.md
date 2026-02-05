# 사전 컴파일된 서브에이전트

> CompiledSubAgent를 사용하여 LangGraph 에이전트를 서브에이전트로 통합하는 방법입니다.

## 개요

`CompiledSubAgent`는 사전 컴파일된 LangGraph 에이전트를 Deep Agents의 서브에이전트로 사용할 수 있게 해줍니다. 이를 통해 복잡한 커스텀 로직, 특수 도구 체인, 또는 기존 에이전트를 재사용할 수 있습니다.

---

## SubAgent vs CompiledSubAgent

### SubAgent (선언적)

```python
from deepagents import create_deep_agent

# 선언적으로 서브에이전트 정의
# Deep Agents가 내부적으로 에이전트 생성
researcher: SubAgent = {
    "name": "researcher",
    "description": "웹 리서치를 수행하는 에이전트",
    "system_prompt": "당신은 리서치 전문가입니다.",
    "tools": [web_search],
    "model": "openai:gpt-4o",
}

agent = create_deep_agent(subagents=[researcher])
```

### CompiledSubAgent (프로그래밍)

```python
from langchain.agents import create_agent

# 사전에 직접 에이전트 컴파일
custom_agent = create_agent(
    model="openai:gpt-4o",
    tools=[web_search, custom_tool],
    system_prompt="커스텀 지시사항...",
)

# 컴파일된 에이전트를 서브에이전트로 등록
compiled: CompiledSubAgent = {
    "name": "custom-agent",
    "description": "커스텀 로직을 가진 에이전트",
    "runnable": custom_agent,  # Runnable 인터페이스 구현
}

agent = create_deep_agent(subagents=[compiled])
```

---

## CompiledSubAgent TypedDict

**소스 위치**: `libs/deepagents/deepagents/middleware/subagents.py:117-147`

```python
from typing_extensions import TypedDict
from langchain_core.runnables import Runnable

class CompiledSubAgent(TypedDict):
    """사전 컴파일된 에이전트 명세

    참고:
        runnable의 상태 스키마에 'messages' 키가 포함되어야 합니다.
        이는 서브에이전트가 메인 에이전트에 결과를 반환하는 데 필요합니다.

    서브에이전트가 완료되면, 'messages' 리스트의 마지막 메시지가
    추출되어 부모 에이전트에 ToolMessage로 반환됩니다.
    """

    name: str
    """서브에이전트의 고유 식별자"""

    description: str
    """서브에이전트가 하는 일 (메인 에이전트가 위임 시점 결정에 사용)"""

    runnable: Runnable
    """사용자 정의 에이전트 구현 (messages 키 필수)"""
```

---

## 필수 요구사항: messages 키

CompiledSubAgent의 `runnable`은 반드시 상태에 `messages` 키를 가져야 합니다.

```python
# 서브에이전트 완료 시 내부 동작
def _return_command_with_state_update(result: dict, tool_call_id: str):
    # messages 키 검증
    if "messages" not in result:
        raise ValueError(
            "CompiledSubAgent must return a state containing a 'messages' key."
        )

    # 마지막 메시지를 ToolMessage로 변환
    message_text = result["messages"][-1].text.rstrip()
    return Command(
        update={
            "messages": [ToolMessage(message_text, tool_call_id=tool_call_id)],
        }
    )
```

---

## 기본 사용법

### 1. LangChain create_agent 사용

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from deepagents import create_deep_agent

# 1. 커스텀 도구 정의
@tool
def calculate(expression: str) -> str:
    """수학 표현식을 계산합니다."""
    try:
        result = eval(expression)
        return f"결과: {result}"
    except Exception as e:
        return f"에러: {e}"

@tool
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """단위를 변환합니다."""
    # 구현...
    pass

# 2. 서브에이전트용 에이전트 컴파일
math_agent = create_agent(
    model="openai:gpt-4o",
    tools=[calculate, unit_convert],
    system_prompt="""당신은 수학 전문가입니다.

## 역할
- 수학 계산 수행
- 단위 변환
- 수학적 분석

## 출력 형식
계산 결과와 풀이 과정을 함께 제공하세요.
""",
)

# 3. CompiledSubAgent로 등록
compiled_math: CompiledSubAgent = {
    "name": "math-expert",
    "description": "수학 계산, 단위 변환, 수학적 분석을 수행하는 전문 에이전트",
    "runnable": math_agent,
}

# 4. 메인 에이전트에 통합
main_agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    subagents=[compiled_math],
)
```

### 2. 커스텀 LangGraph 그래프 사용

```python
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage

# 1. 상태 스키마 정의 (messages 키 필수!)
class CustomState(TypedDict):
    messages: list[BaseMessage]  # 필수!
    context: str
    step_count: int

# 2. 노드 함수 정의
def analyze_node(state: CustomState) -> dict:
    """분석 노드"""
    messages = state["messages"]
    # 분석 로직...
    return {
        "messages": [AIMessage(content="분석 완료")],
        "context": "analyzed",
    }

def synthesize_node(state: CustomState) -> dict:
    """합성 노드"""
    # 합성 로직...
    return {
        "messages": [AIMessage(content="최종 결과: ...")],
    }

# 3. 그래프 구성
builder = StateGraph(CustomState)
builder.add_node("analyze", analyze_node)
builder.add_node("synthesize", synthesize_node)
builder.add_edge("analyze", "synthesize")
builder.set_entry_point("analyze")
builder.set_finish_point("synthesize")

# 4. 그래프 컴파일
custom_graph = builder.compile()

# 5. CompiledSubAgent로 등록
compiled_custom: CompiledSubAgent = {
    "name": "custom-workflow",
    "description": "분석-합성 워크플로우를 수행하는 에이전트",
    "runnable": custom_graph,
}
```

---

## 고급 패턴

### 멀티스텝 워크플로우 에이전트

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing_extensions import TypedDict

class ResearchState(TypedDict):
    messages: list[BaseMessage]
    query: str
    sources: list[dict]
    analysis: str
    final_report: str

def plan_research(state: ResearchState) -> dict:
    """연구 계획 수립"""
    model = ChatOpenAI(model="gpt-4o")
    query = state["messages"][-1].content

    response = model.invoke([
        HumanMessage(content=f"다음 주제에 대한 연구 계획을 세워주세요: {query}")
    ])

    return {
        "query": query,
        "messages": [AIMessage(content=f"연구 계획 수립 완료: {response.content}")],
    }

def gather_sources(state: ResearchState) -> dict:
    """소스 수집"""
    # 웹 검색, 데이터베이스 조회 등
    sources = [
        {"title": "Source 1", "url": "...", "content": "..."},
        {"title": "Source 2", "url": "...", "content": "..."},
    ]

    return {
        "sources": sources,
        "messages": [AIMessage(content=f"{len(sources)}개 소스 수집 완료")],
    }

def analyze_sources(state: ResearchState) -> dict:
    """소스 분석"""
    model = ChatOpenAI(model="gpt-4o")
    sources = state["sources"]

    # 각 소스 분석
    analysis = model.invoke([
        HumanMessage(content=f"다음 소스들을 분석해주세요:\n{sources}")
    ])

    return {
        "analysis": analysis.content,
        "messages": [AIMessage(content="소스 분석 완료")],
    }

def write_report(state: ResearchState) -> dict:
    """최종 보고서 작성"""
    model = ChatOpenAI(model="gpt-4o")

    report = model.invoke([
        HumanMessage(content=f"""
다음 정보를 바탕으로 보고서를 작성해주세요:

주제: {state['query']}
분석: {state['analysis']}
소스: {state['sources']}
""")
    ])

    return {
        "final_report": report.content,
        "messages": [AIMessage(content=report.content)],  # 최종 결과
    }

# 그래프 구성
builder = StateGraph(ResearchState)
builder.add_node("plan", plan_research)
builder.add_node("gather", gather_sources)
builder.add_node("analyze", analyze_sources)
builder.add_node("report", write_report)

builder.add_edge("plan", "gather")
builder.add_edge("gather", "analyze")
builder.add_edge("analyze", "report")
builder.add_edge("report", END)

builder.set_entry_point("plan")

research_workflow = builder.compile()

# CompiledSubAgent로 등록
deep_research: CompiledSubAgent = {
    "name": "deep-researcher",
    "description": "심층 리서치를 수행하는 에이전트. "
                   "계획 수립 → 소스 수집 → 분석 → 보고서 작성 워크플로우 실행.",
    "runnable": research_workflow,
}
```

### 조건부 분기 에이전트

```python
from langgraph.graph import StateGraph, END

class RouterState(TypedDict):
    messages: list[BaseMessage]
    task_type: str
    result: str

def classify_task(state: RouterState) -> dict:
    """작업 유형 분류"""
    message = state["messages"][-1].content

    # 간단한 분류 로직
    if "코드" in message or "프로그래밍" in message:
        task_type = "coding"
    elif "분석" in message or "데이터" in message:
        task_type = "analysis"
    else:
        task_type = "general"

    return {"task_type": task_type}

def handle_coding(state: RouterState) -> dict:
    """코딩 작업 처리"""
    return {
        "result": "코딩 결과...",
        "messages": [AIMessage(content="코딩 작업 완료")],
    }

def handle_analysis(state: RouterState) -> dict:
    """분석 작업 처리"""
    return {
        "result": "분석 결과...",
        "messages": [AIMessage(content="분석 작업 완료")],
    }

def handle_general(state: RouterState) -> dict:
    """일반 작업 처리"""
    return {
        "result": "일반 결과...",
        "messages": [AIMessage(content="일반 작업 완료")],
    }

def route_task(state: RouterState) -> str:
    """작업 라우팅"""
    task_type = state.get("task_type", "general")
    return {
        "coding": "coding",
        "analysis": "analysis",
        "general": "general",
    }.get(task_type, "general")

# 그래프 구성
builder = StateGraph(RouterState)
builder.add_node("classify", classify_task)
builder.add_node("coding", handle_coding)
builder.add_node("analysis", handle_analysis)
builder.add_node("general", handle_general)

builder.add_conditional_edges(
    "classify",
    route_task,
    {
        "coding": "coding",
        "analysis": "analysis",
        "general": "general",
    }
)

for node in ["coding", "analysis", "general"]:
    builder.add_edge(node, END)

builder.set_entry_point("classify")

router_agent = builder.compile()

# CompiledSubAgent로 등록
smart_router: CompiledSubAgent = {
    "name": "smart-router",
    "description": "작업 유형을 자동으로 분류하고 적절한 처리기로 라우팅하는 에이전트",
    "runnable": router_agent,
}
```

---

## 서브에이전트 상태 격리

서브에이전트는 메인 에이전트와 격리된 컨텍스트에서 실행됩니다.

### 제외되는 상태 키

```python
_EXCLUDED_STATE_KEYS = {
    "messages",           # 새 히스토리 시작
    "todos",              # 자체 할 일 관리
    "structured_response",
    "skills_metadata",    # 자체 스킬 로드
    "memory_contents",    # 자체 메모리 로드
}
```

### 상태 전달 흐름

```
메인 에이전트 상태
├─ messages: [...]  ← 제외 (서브에이전트는 새 히스토리로 시작)
├─ todos: [...]     ← 제외
├─ files: {...}     ← 전달됨 (파일 시스템 공유)
├─ custom_key: ...  ← 전달됨
└─ ...

↓ task() 호출

서브에이전트 상태
├─ messages: [HumanMessage(description)]  ← 새로 생성
├─ files: {...}     ← 메인에서 상속
├─ custom_key: ...  ← 메인에서 상속
└─ ...

↓ 실행 완료

메인 에이전트에 반환
└─ messages: [ToolMessage(마지막 메시지)]  ← 마지막 응답만 반환
```

---

## 기존 에이전트 통합

### 다른 프레임워크 에이전트 래핑

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage

class ExternalAgentWrapper:
    """외부 에이전트를 LangGraph Runnable로 래핑"""

    def __init__(self, external_agent):
        self.external_agent = external_agent

    def invoke(self, state: dict) -> dict:
        """외부 에이전트 호출"""
        # 메시지에서 작업 추출
        task = state["messages"][-1].content

        # 외부 에이전트 호출
        result = self.external_agent.run(task)

        # messages 키를 포함한 상태 반환
        return {
            "messages": [AIMessage(content=result)],
        }

    async def ainvoke(self, state: dict) -> dict:
        """비동기 호출"""
        return self.invoke(state)

# 사용
from some_framework import SomeAgent

external = SomeAgent(config={...})
wrapped = ExternalAgentWrapper(external)

compiled_external: CompiledSubAgent = {
    "name": "external-agent",
    "description": "외부 프레임워크 에이전트",
    "runnable": RunnableLambda(wrapped.invoke),
}
```

### Runnable 인터페이스 구현

```python
from langchain_core.runnables import Runnable

class MyCustomRunnable(Runnable):
    """커스텀 Runnable 구현"""

    def invoke(self, input: dict, config=None) -> dict:
        """동기 실행"""
        messages = input.get("messages", [])
        task = messages[-1].content if messages else ""

        # 커스텀 로직
        result = self._process(task)

        return {
            "messages": [AIMessage(content=result)],
        }

    async def ainvoke(self, input: dict, config=None) -> dict:
        """비동기 실행"""
        return self.invoke(input, config)

    def _process(self, task: str) -> str:
        """실제 처리 로직"""
        # 구현...
        return f"처리 완료: {task}"
```

---

## 테스트

```python
import pytest
from langchain_core.messages import HumanMessage

def test_compiled_subagent_messages_key():
    """messages 키 요구사항 테스트"""

    # 유효한 그래프 (messages 키 포함)
    def valid_node(state):
        return {"messages": [AIMessage(content="결과")]}

    builder = StateGraph({"messages": list})
    builder.add_node("process", valid_node)
    builder.set_entry_point("process")
    valid_graph = builder.compile()

    compiled = CompiledSubAgent(
        name="test",
        description="테스트",
        runnable=valid_graph,
    )

    result = valid_graph.invoke({"messages": [HumanMessage(content="테스트")]})
    assert "messages" in result

def test_compiled_subagent_in_main_agent():
    """메인 에이전트와 통합 테스트"""
    from deepagents import create_deep_agent

    # 간단한 서브에이전트
    def echo_node(state):
        msg = state["messages"][-1].content
        return {"messages": [AIMessage(content=f"Echo: {msg}")]}

    builder = StateGraph({"messages": list})
    builder.add_node("echo", echo_node)
    builder.set_entry_point("echo")
    echo_graph = builder.compile()

    compiled = {
        "name": "echo-agent",
        "description": "메시지를 에코하는 에이전트",
        "runnable": echo_graph,
    }

    # 메인 에이전트에 통합
    main = create_deep_agent(subagents=[compiled])

    # task() 도구로 호출되는지 확인
    assert any(t.name == "task" for t in main.get_tools())
```

---

## 모범 사례

### 1. 명확한 상태 스키마

```python
# ✅ 좋은 예: 명확한 타입 정의
class MyState(TypedDict):
    messages: list[BaseMessage]  # 필수!
    query: str
    results: list[dict]

# ❌ 나쁜 예: Any 타입 남용
class BadState(TypedDict):
    messages: Any
    data: Any
```

### 2. 최종 메시지에 결과 포함

```python
# ✅ 좋은 예: 마지막 메시지에 유용한 결과
def final_node(state):
    report = generate_report(state)
    return {
        "messages": [AIMessage(content=report)],  # 이것만 메인에 반환됨
    }

# ❌ 나쁜 예: 빈 메시지
def bad_final_node(state):
    return {
        "messages": [AIMessage(content="완료")],  # 정보 없음
        "actual_result": "..."  # 이건 반환 안 됨!
    }
```

### 3. 에러 처리

```python
def safe_node(state):
    try:
        result = risky_operation()
        return {"messages": [AIMessage(content=result)]}
    except Exception as e:
        # 에러도 메시지로 반환
        return {"messages": [AIMessage(content=f"에러 발생: {e}")]}
```

---

## 관련 문서

- [서브에이전트 오케스트레이션](../02-core-concepts/subagent-orchestration.md)
- [서브에이전트 API](../05-api-reference/subagent-api.md)
- [병렬 서브에이전트 패턴](../04-patterns/parallel-subagents.md)
