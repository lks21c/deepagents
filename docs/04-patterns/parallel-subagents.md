# 병렬 서브에이전트 패턴

> 여러 서브에이전트를 동시에 실행하여 처리 시간을 단축하는 패턴입니다.

## 개요

병렬 서브에이전트 패턴은 독립적인 작업을 여러 서브에이전트에게 동시에 위임하여 전체 처리 시간을 줄입니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Parallel SubAgent Execution                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                    ┌──────────────────┐                             │
│                    │   Main Agent     │                             │
│                    └────────┬─────────┘                             │
│                             │                                        │
│           ┌─────────────────┼─────────────────┐                     │
│           │                 │                 │                     │
│           ▼                 ▼                 ▼                     │
│    ┌────────────┐    ┌────────────┐    ┌────────────┐              │
│    │ task()     │    │ task()     │    │ task()     │              │
│    │ Player A   │    │ Player B   │    │ Player C   │              │
│    └─────┬──────┘    └─────┬──────┘    └─────┬──────┘              │
│          │                 │                 │                      │
│          │    ═══════════════════════════    │                      │
│          │         병렬 실행 (Parallel)       │                      │
│          │    ═══════════════════════════    │                      │
│          │                 │                 │                      │
│          ▼                 ▼                 ▼                      │
│    ┌────────────┐    ┌────────────┐    ┌────────────┐              │
│    │ Result A   │    │ Result B   │    │ Result C   │              │
│    └─────┬──────┘    └─────┬──────┘    └─────┬──────┘              │
│          │                 │                 │                      │
│          └─────────────────┼─────────────────┘                      │
│                            │                                        │
│                            ▼                                        │
│                    ┌──────────────────┐                             │
│                    │   Main Agent     │                             │
│                    │   (통합 분석)     │                             │
│                    └──────────────────┘                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 기본 구현

### 시스템 프롬프트 설정

**소스 참조**: `libs/deepagents/deepagents/middleware/subagents.py:137-138`

```python
MAIN_INSTRUCTIONS = """
## 병렬 실행 지침

1. Launch multiple agents concurrently whenever possible,
   to maximize performance; to do that, use a single message
   with multiple tool uses

2. 독립적인 작업은 병렬로 위임하세요
3. 의존성이 있는 작업은 순차적으로 처리하세요
"""
```

### 에이전트 설정

```python
from deepagents import create_deep_agent

research_agent = {
    "name": "researcher",
    "description": "웹 리서치를 수행하는 에이전트",
    "system_prompt": "당신은 리서치 전문가입니다...",
    "tools": [web_search, think],
}

agent = create_deep_agent(
    system_prompt=MAIN_INSTRUCTIONS,
    subagents=[research_agent],
)
```

---

## 병렬 실행 예시

### 여러 주제 동시 리서치

사용자 요청:
> "마이클 조던, 르브론 제임스, 스테판 커리의 커리어 통계를 비교해주세요"

메인 에이전트 응답:

```python
# 단일 응답에서 3개의 task() 호출 동시 생성
[
    task(
        description="마이클 조던의 NBA 커리어 통계를 조사해주세요",
        subagent_type="researcher"
    ),
    task(
        description="르브론 제임스의 NBA 커리어 통계를 조사해주세요",
        subagent_type="researcher"
    ),
    task(
        description="스테판 커리의 NBA 커리어 통계를 조사해주세요",
        subagent_type="researcher"
    ),
]
```

**실행 시간 비교**:
- 순차 실행: ~90초 (30초 × 3)
- 병렬 실행: ~35초 (최대 소요 시간 + 오버헤드)

---

## 병렬 vs 순차 결정 기준

### 병렬 실행이 적합한 경우

| 조건 | 예시 |
|------|------|
| 독립적인 작업 | 서로 다른 주제 리서치 |
| 데이터 의존성 없음 | 각 선수의 통계 조회 |
| 결과 순서 무관 | 여러 파일 동시 분석 |
| 시간이 중요 | 빠른 응답 필요 |

### 순차 실행이 필요한 경우

| 조건 | 예시 |
|------|------|
| 의존성 있음 | A 결과가 B 입력으로 필요 |
| 상태 공유 | 공유 리소스 접근 |
| 순서 중요 | 단계별 처리 |
| 오류 전파 | 이전 단계 실패 시 중단 |

---

## 구현 패턴

### 패턴 1: 데이터 수집 후 통합

```python
# 시스템 프롬프트
"""
## 작업 방식

1. 데이터 수집 단계 (병렬)
   - 각 데이터 소스에 대해 task() 동시 호출
   - 모든 결과 수집까지 대기

2. 분석 단계 (순차)
   - 수집된 데이터 통합
   - 비교 분석 수행
   - 최종 보고서 작성
"""
```

### 패턴 2: 팬아웃/팬인 (Fan-out/Fan-in)

```python
# 메인 에이전트가 자동으로 수행
"""
1. 팬아웃: 작업을 여러 서브에이전트에 분배
   task("분석 A") | task("분석 B") | task("분석 C")

2. 팬인: 모든 결과를 수집하여 통합
   → 결과 A + 결과 B + 결과 C → 최종 분석
"""
```

### 패턴 3: 조건부 병렬화

```python
# 시스템 프롬프트
"""
## 병렬화 규칙

- 리서치 작업: 최대 3개 동시 실행
- 코드 분석: 최대 5개 동시 실행 (가벼운 작업)
- 파일 생성: 순차 실행 (충돌 방지)
"""
```

---

## 제한사항 및 최적화

### 동시 실행 제한

```python
# 권장 설정
MAX_CONCURRENT_RESEARCH_UNITS = 3  # API 제한 고려
MAX_RESEARCHER_ITERATIONS = 3      # 각 리서치당 최대 검색 횟수
```

**제한 이유**:
- API 요금 관리
- 품질 저하 방지
- 리소스 효율화

### 오류 처리

```python
"""
## 오류 처리 지침

1. 일부 서브에이전트 실패 시:
   - 성공한 결과로 부분 응답 생성
   - 실패한 작업 재시도 고려

2. 전체 실패 시:
   - 사용자에게 알림
   - 대안 제시
"""
```

---

## 실제 구현 예제

### 멀티 소스 리서치 에이전트

```python
from deepagents import create_deep_agent
from langchain_core.tools import tool

@tool
def web_search(query: str) -> dict:
    """웹에서 정보를 검색합니다."""
    # 검색 구현
    pass

@tool
def think(thought: str) -> str:
    """생각을 정리합니다."""
    return f"Thought recorded: {thought}"

# 리서치 서브에이전트
research_agent = {
    "name": "researcher",
    "description": "웹 리서치를 수행하는 전문 에이전트",
    "system_prompt": """당신은 철저한 리서치를 수행하는 연구원입니다.

## 작업 방식
1. think 도구로 검색 전략 수립
2. web_search로 정보 수집
3. 수집한 정보 검증 및 정리
4. 구조화된 결과 반환
""",
    "tools": [web_search, think],
}

# 메인 에이전트
PARALLEL_INSTRUCTIONS = """당신은 리서치 프로젝트를 관리합니다.

## 병렬 실행 규칙

독립적인 주제를 리서치할 때:
1. 각 주제에 대해 별도의 task() 호출 생성
2. 단일 응답에서 모든 task() 호출을 포함 (병렬 실행)
3. 모든 결과를 받은 후 통합 분석

예시:
- "A와 B를 비교해주세요" → task(A), task(B) 병렬 호출
- "A를 먼저 조사하고, 그 결과를 바탕으로 B 조사" → 순차 호출
"""

agent = create_deep_agent(
    system_prompt=PARALLEL_INSTRUCTIONS,
    subagents=[research_agent],
    tools=[web_search, think],
)

# 실행
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Tesla, BYD, Rivian의 전기차 기술을 비교해주세요"
    }]
})
```

---

## 성능 모니터링

### 실행 시간 측정

```python
import time

async def timed_invoke(agent, messages):
    start = time.time()

    async for chunk in agent.astream(
        {"messages": messages},
        stream_mode="values",
    ):
        # 진행 상황 모니터링
        if "messages" in chunk:
            latest = chunk["messages"][-1]
            if hasattr(latest, "tool_calls") and latest.tool_calls:
                for tc in latest.tool_calls:
                    if tc.get("name") == "task":
                        print(f"서브에이전트 시작: {tc.get('args', {}).get('description', '')[:30]}...")

    elapsed = time.time() - start
    print(f"총 실행 시간: {elapsed:.2f}초")
```

---

## 모범 사례

### 1. 명확한 작업 분리

```markdown
✅ 좋은 예:
- task("Tesla의 배터리 기술 조사")
- task("BYD의 배터리 기술 조사")

❌ 나쁜 예:
- task("Tesla와 BYD의 배터리 기술 비교")  # 하나의 태스크에 두 주제
```

### 2. 적절한 세분화

```markdown
✅ 적절한 세분화:
- 3-5개의 병렬 서브에이전트

❌ 과도한 세분화:
- 10개 이상의 서브에이전트 (관리 오버헤드)
```

### 3. 결과 통합 계획

```markdown
병렬 실행 전 통합 계획 수립:
1. 어떤 필드를 비교할지 결정
2. 결과 형식 표준화
3. 누락 데이터 처리 방안
```

---

## 다음 단계

- [컨텍스트 관리](./context-management.md)
- [Human-in-the-Loop 패턴](./human-in-the-loop.md)
- [서브에이전트 오케스트레이션](../02-core-concepts/subagent-orchestration.md)
