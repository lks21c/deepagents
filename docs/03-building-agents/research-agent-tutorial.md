# 리서치 에이전트 튜토리얼

> 웹 리서치를 수행하는 전문 에이전트를 단계별로 구축합니다.

## 개요

이 튜토리얼에서는 다음 기능을 가진 리서치 에이전트를 구축합니다:

- 웹 검색 (Tavily API 사용)
- 생각 정리 (Think Tool)
- 메인 에이전트와 리서치 서브에이전트 분리
- 병렬 리서치 지원

**소스 참조**: `examples/deep_research/agent.py`

## 사전 요구사항

```bash
pip install deepagents tavily-python
```

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export TAVILY_API_KEY="your-tavily-key"
```

## 프로젝트 구조

```
research_agent/
├── agent.py           # 메인 에이전트 정의
├── tools.py           # 도구 정의
└── prompts.py         # 프롬프트 정의
```

---

## Step 1: 도구 정의

### tools.py

```python
"""리서치 에이전트용 도구 정의"""

import os
from langchain_core.tools import tool


@tool
def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> dict:
    """웹에서 정보를 검색합니다.

    Tavily Search API를 사용하여 실시간 웹 검색을 수행합니다.

    Args:
        query: 검색 쿼리. 구체적이고 명확하게 작성하세요.
        max_results: 반환할 최대 결과 수 (기본값: 5)
        search_depth: 검색 깊이 - "basic" 또는 "advanced" (기본값: basic)

    Returns:
        검색 결과 딕셔너리:
        - results: 검색 결과 목록
        - answer: AI가 생성한 요약 답변 (있는 경우)

    Raises:
        ValueError: API 키가 설정되지 않은 경우
    """
    from tavily import TavilyClient

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {"error": "TAVILY_API_KEY environment variable not set"}

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
        )
        return response
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


@tool
def think(thought: str) -> str:
    """생각을 정리하고 기록합니다.

    복잡한 문제를 해결할 때 중간 생각을 정리하는 데 사용합니다.
    검색 전 전략 수립, 정보 분석, 결론 도출 시 활용하세요.

    Args:
        thought: 정리할 생각 내용

    Returns:
        확인 메시지
    """
    # 실제로는 아무것도 하지 않지만,
    # LLM이 자신의 사고 과정을 명시적으로 기록하게 함
    return f"Thought recorded: {thought}"
```

**코드 설명**:

| 라인 | 설명 |
|------|------|
| `@tool` 데코레이터 | 함수를 LangChain 도구로 변환 |
| docstring | LLM이 도구 사용법을 이해하는 데 필수 |
| `max_results` | 결과 수 제한으로 토큰 사용량 조절 |
| `search_depth` | 상세 검색 필요 시 "advanced" 사용 |
| `think` 도구 | LLM의 추론 과정을 명시화 |

---

## Step 2: 프롬프트 정의

### prompts.py

```python
"""리서치 에이전트용 프롬프트 정의"""

from datetime import datetime

# 현재 날짜 가져오기
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")


RESEARCHER_INSTRUCTIONS = """당신은 철저하고 체계적인 리서치를 수행하는 연구원입니다.

## 현재 날짜
{date}

## 작업 원칙

### 1. 검색 전략
- think 도구로 먼저 검색 전략을 수립하세요
- 다양한 각도에서 쿼리를 작성하세요
- 일반적인 것에서 구체적인 것으로 진행하세요

### 2. 정보 수집
- 최소 3개 이상의 다른 검색 쿼리 사용
- 검색 결과를 비판적으로 평가하세요
- 신뢰할 수 있는 출처 우선

### 3. 정보 정리
- 핵심 발견사항을 요약하세요
- 출처를 명확히 기록하세요
- 상충되는 정보가 있으면 모두 보고하세요

### 4. 결과 형식
- 핵심 요약 (3-5줄)
- 상세 내용 (섹션별 정리)
- 출처 목록
- 추가 조사 필요 사항 (있는 경우)

## 주의사항
- 검색 결과에 없는 정보를 추측하지 마세요
- 불확실한 정보는 "확인 필요"로 표시하세요
- 편향된 출처는 그 사실을 언급하세요
"""


RESEARCH_WORKFLOW_INSTRUCTIONS = """당신은 리서치 프로젝트를 관리하는 오케스트레이터입니다.

## 역할
- 리서치 요청을 분석하고 계획 수립
- 리서치 작업을 researcher 서브에이전트에 위임
- 결과를 종합하여 최종 보고서 작성

## 작업 흐름

### 1. 요청 분석
사용자의 리서치 요청을 분석하여:
- 핵심 질문 파악
- 필요한 리서치 범위 결정
- 서브 주제로 분해 (필요시)

### 2. 리서치 위임
- 각 주제를 researcher 서브에이전트에 위임
- 독립적인 주제는 병렬로 위임 가능
- 한 번에 하나의 주제만 위임

### 3. 결과 통합
- 각 리서치 결과를 종합
- 일관성 검토
- 최종 보고서 작성
"""


SUBAGENT_DELEGATION_INSTRUCTIONS = """## 서브에이전트 사용 지침

### researcher 서브에이전트
- 웹 리서치가 필요한 작업에 사용
- 한 번에 하나의 명확한 주제만 위임
- 병렬 리서치: 최대 {max_concurrent_research_units}개 동시 실행

### 위임 예시
```
task(
    description="2024년 AI 트렌드 중 LLM 분야를 조사해주세요.
    주요 발전사항, 대표 모델, 향후 전망을 포함해주세요.",
    subagent_type="research-agent"
)
```

### 제한사항
- 리서치 단위당 최대 {max_researcher_iterations}회 검색
- 너무 넓은 주제는 세분화하여 위임
"""
```

**설계 이유**:

| 항목 | 이유 |
|------|------|
| 날짜 포함 | 시간에 민감한 정보 검색 정확도 향상 |
| 작업 원칙 | LLM이 체계적으로 작업하도록 유도 |
| 결과 형식 | 일관된 출력 품질 보장 |
| 위임 지침 | 메인-서브 에이전트 간 효율적 협업 |

---

## Step 3: 에이전트 정의

### agent.py

```python
"""리서치 에이전트 - Deep Agents 기반 웹 리서치 에이전트"""

from datetime import datetime

from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

from tools import tavily_search, think_tool
from prompts import (
    RESEARCHER_INSTRUCTIONS,
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)

# === 설정 ===

# 동시 리서치 단위 제한
MAX_CONCURRENT_RESEARCH_UNITS = 3

# 리서치 단위당 최대 검색 반복 횟수
MAX_RESEARCHER_ITERATIONS = 3

# 현재 날짜
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")


# === 메인 에이전트 프롬프트 ===

MAIN_INSTRUCTIONS = (
    RESEARCH_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_research_units=MAX_CONCURRENT_RESEARCH_UNITS,
        max_researcher_iterations=MAX_RESEARCHER_ITERATIONS,
    )
)


# === 리서치 서브에이전트 정의 ===

research_sub_agent = {
    "name": "research-agent",
    "description": (
        "웹 리서치를 수행하는 전문 에이전트. "
        "한 번에 하나의 주제만 위임하세요. "
        "복잡한 주제는 세분화하여 여러 번 호출하세요."
    ),
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=CURRENT_DATE),
    "tools": [tavily_search, think_tool],
}


# === 모델 설정 ===

# Claude Sonnet 4.5 사용
model = init_chat_model(
    model="anthropic:claude-sonnet-4-5-20250929",
    temperature=0.0,  # 일관된 결과를 위해 0으로 설정
)


# === 에이전트 생성 ===

agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think_tool],  # 메인도 직접 검색 가능
    system_prompt=MAIN_INSTRUCTIONS,
    subagents=[research_sub_agent],
)


# === 실행 함수 ===

def research(query: str) -> str:
    """리서치를 수행하고 결과를 반환합니다.

    Args:
        query: 리서치할 주제

    Returns:
        리서치 결과 문자열
    """
    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content


# === CLI 인터페이스 ===

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "2024년 AI 에이전트 기술 트렌드에 대해 조사해주세요"

    print(f"🔍 리서치 주제: {query}\n")
    print("-" * 50)

    result = research(query)
    print(result)
```

**코드 상세 설명**:

```python
# 동시 리서치 제한
MAX_CONCURRENT_RESEARCH_UNITS = 3
```
- **이유**: 너무 많은 병렬 요청은 API 제한에 걸리거나 품질이 저하될 수 있음
- **값 선택**: 3개는 속도와 품질의 균형점

```python
research_sub_agent = {
    "name": "research-agent",
    ...
    "tools": [tavily_search, think_tool],
}
```
- **name**: `task()` 호출 시 `subagent_type`으로 사용
- **description**: 메인 에이전트가 위임 결정에 사용
- **tools**: 메인과 별도로 도구 세트 지정 가능

```python
model = init_chat_model(
    model="anthropic:claude-sonnet-4-5-20250929",
    temperature=0.0,
)
```
- **temperature=0.0**: 리서치 작업에서 일관된 결과를 위해 낮은 값 사용

---

## Step 4: 실행

```bash
# 기본 쿼리로 실행
python agent.py

# 커스텀 쿼리
python agent.py "양자 컴퓨팅의 현재 상용화 수준을 조사해주세요"
```

### 예상 출력

```
🔍 리서치 주제: 2024년 AI 에이전트 기술 트렌드에 대해 조사해주세요

--------------------------------------------------

## 핵심 요약

2024년 AI 에이전트 기술은 다음 세 가지 주요 트렌드를 보이고 있습니다:
1. 멀티모달 에이전트의 부상
2. 도구 사용 능력의 고도화
3. 자율 에이전트 프레임워크의 성숙

## 상세 내용

### 1. 멀티모달 에이전트
...

### 2. 도구 사용 (Tool Use)
...

### 3. 자율 에이전트 프레임워크
...

## 출처
- [출처 1 URL]
- [출처 2 URL]
...

## 추가 조사 필요
- 특정 산업별 적용 사례
- 규제 및 윤리적 고려사항
```

---

## 고급 기능

### 비동기 스트리밍

```python
import asyncio

async def stream_research(query: str):
    """실시간으로 리서치 진행 상황을 출력합니다."""
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        if "messages" in chunk:
            latest = chunk["messages"][-1]

            # 도구 호출 표시
            if hasattr(latest, "tool_calls") and latest.tool_calls:
                for tc in latest.tool_calls:
                    name = tc.get("name", "unknown")
                    if name == "task":
                        print(f"🔄 리서치 위임: {tc.get('args', {}).get('description', '')[:50]}...")
                    elif name == "tavily_search":
                        print(f"🔍 검색 중: {tc.get('args', {}).get('query', '')}")

            # 최종 응답 출력
            if hasattr(latest, "content") and latest.content:
                print(latest.content)

asyncio.run(stream_research("양자 컴퓨팅 트렌드"))
```

### 체크포인터로 세션 유지

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think_tool],
    system_prompt=MAIN_INSTRUCTIONS,
    subagents=[research_sub_agent],
    checkpointer=MemorySaver(),
)

# 첫 번째 리서치
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "AI 에이전트 트렌드를 조사해주세요"}]},
    config={"configurable": {"thread_id": "research-session-1"}}
)

# 추가 질문 (이전 컨텍스트 유지)
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "그 중 LangChain에 대해 더 자세히 알려주세요"}]},
    config={"configurable": {"thread_id": "research-session-1"}}
)
```

---

## 문제 해결

### 검색 결과 없음

```python
# 검색 깊이 증가
response = client.search(query, search_depth="advanced")
```

### API 속도 제한

```python
import time

# 요청 간 지연 추가
@tool
def tavily_search_with_delay(query: str) -> dict:
    time.sleep(1)  # 1초 대기
    return tavily_search(query)
```

### 토큰 한도 초과

```python
# max_results 줄이기
@tool
def tavily_search(query: str, max_results: int = 3) -> dict:
    ...
```

---

## 다음 단계

- [SQL 에이전트 튜토리얼](./sql-agent-tutorial.md)
- [콘텐츠 에이전트 튜토리얼](./content-agent-tutorial.md)
- [병렬 서브에이전트 패턴](../04-patterns/parallel-subagents.md)
