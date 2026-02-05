# 커스텀 도구 패턴

> 에이전트에 사용자 정의 도구를 추가하는 패턴입니다.

## 개요

Deep Agents는 기본 도구 외에 커스텀 도구를 추가할 수 있습니다. LangChain의 `@tool` 데코레이터를 사용하여 함수를 도구로 변환합니다.

---

## 기본 도구 생성

### @tool 데코레이터

```python
from langchain_core.tools import tool

@tool
def calculate(expression: str) -> float:
    """수학 표현식을 계산합니다.

    Args:
        expression: 계산할 수식 (예: "2 + 3 * 4")

    Returns:
        계산 결과
    """
    return eval(expression)

# 에이전트에 추가
agent = create_deep_agent(
    tools=[calculate],
)
```

### Docstring 중요성

LLM은 docstring을 읽고 도구 사용법을 이해합니다.

```python
# ✅ 좋은 예: 상세한 docstring
@tool
def web_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> dict:
    """웹에서 정보를 검색합니다.

    Tavily Search API를 사용하여 실시간 웹 검색을 수행합니다.

    Args:
        query: 검색 쿼리. 구체적이고 명확하게 작성하세요.
        max_results: 반환할 최대 결과 수 (기본값: 5)
        search_depth: 검색 깊이 - "basic" 또는 "advanced"

    Returns:
        검색 결과 딕셔너리:
        - results: 검색 결과 목록
        - answer: AI가 생성한 요약 답변

    Raises:
        ValueError: API 키가 설정되지 않은 경우
    """
    # 구현
    pass

# ❌ 나쁜 예: 불충분한 docstring
@tool
def web_search(query: str) -> dict:
    """검색합니다."""  # LLM이 이해하기 어려움
    pass
```

---

## 런타임 접근

### ToolRuntime 사용

도구에서 에이전트 상태에 접근하려면 `ToolRuntime`을 사용합니다.

```python
from langchain.tools import ToolRuntime

@tool
def get_current_todos(runtime: ToolRuntime) -> list:
    """현재 할 일 목록을 반환합니다.

    Args:
        runtime: 도구 런타임 컨텍스트 (자동 주입)

    Returns:
        현재 할 일 목록
    """
    return runtime.state.get("todos", [])
```

### 상태 업데이트

도구에서 상태를 업데이트하려면 `Command`를 반환합니다.

```python
from langgraph.types import Command
from langchain_core.messages import ToolMessage

@tool
def add_note(runtime: ToolRuntime, note: str) -> Command:
    """노트를 추가합니다.

    Args:
        runtime: 도구 런타임 컨텍스트
        note: 추가할 노트 내용

    Returns:
        상태 업데이트 Command
    """
    current_notes = runtime.state.get("notes", [])
    updated_notes = [*current_notes, note]

    return Command(
        update={
            "notes": updated_notes,
            "messages": [
                ToolMessage(
                    content=f"노트 추가됨: {note}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
```

---

## 비동기 도구

### 비동기 함수

```python
import aiohttp

@tool
async def fetch_url(url: str) -> str:
    """URL에서 콘텐츠를 가져옵니다.

    Args:
        url: 가져올 URL

    Returns:
        페이지 콘텐츠
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

### 병렬 요청

```python
import asyncio

@tool
async def fetch_multiple_urls(urls: list[str]) -> list[str]:
    """여러 URL을 병렬로 가져옵니다.

    Args:
        urls: URL 목록

    Returns:
        각 URL의 콘텐츠 목록
    """
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [await r.text() for r in responses]
```

---

## 외부 API 통합

### Tavily 검색

```python
import os
from tavily import TavilyClient

@tool
def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> dict:
    """Tavily API로 웹 검색을 수행합니다.

    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수
        search_depth: "basic" 또는 "advanced"

    Returns:
        검색 결과 딕셔너리
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {"error": "TAVILY_API_KEY not set"}

    try:
        client = TavilyClient(api_key=api_key)
        return client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
        )
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}
```

### 데이터베이스 쿼리

```python
import sqlite3

@tool
def query_database(sql: str, params: list | None = None) -> list[dict]:
    """SQLite 데이터베이스에서 쿼리를 실행합니다.

    Args:
        sql: 실행할 SQL 쿼리 (SELECT만 허용)
        params: 쿼리 파라미터 (선택)

    Returns:
        쿼리 결과 목록

    Raises:
        ValueError: SELECT가 아닌 쿼리 시도 시
    """
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("SELECT 쿼리만 허용됩니다")

    conn = sqlite3.connect("app.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if params:
        cursor.execute(sql, params)
    else:
        cursor.execute(sql)

    results = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return results
```

---

## 도구 조합

### 도구 체이닝

```python
@tool
def research_and_summarize(topic: str) -> str:
    """주제를 검색하고 요약합니다.

    내부적으로 웹 검색을 수행한 후 결과를 요약합니다.

    Args:
        topic: 조사할 주제

    Returns:
        요약된 리서치 결과
    """
    # 1. 검색
    search_results = tavily_search(topic, max_results=10)

    # 2. 결과 추출
    contents = [r.get("content", "") for r in search_results.get("results", [])]

    # 3. 요약 (간단한 예시)
    combined = "\n".join(contents[:5])
    return f"## {topic}에 대한 리서치 요약\n\n{combined[:2000]}..."
```

### 조건부 도구 선택

```python
@tool
def smart_search(
    query: str,
    search_type: str = "auto",
) -> dict:
    """쿼리 유형에 따라 적절한 검색을 수행합니다.

    Args:
        query: 검색 쿼리
        search_type: "web", "academic", "news", 또는 "auto"

    Returns:
        검색 결과
    """
    if search_type == "auto":
        # 쿼리 분석으로 유형 결정
        if any(kw in query.lower() for kw in ["논문", "연구", "학술"]):
            search_type = "academic"
        elif any(kw in query.lower() for kw in ["뉴스", "최근", "오늘"]):
            search_type = "news"
        else:
            search_type = "web"

    if search_type == "academic":
        return search_academic(query)
    elif search_type == "news":
        return search_news(query)
    else:
        return tavily_search(query)
```

---

## 에러 처리

### 명확한 에러 메시지

```python
@tool
def process_file(file_path: str) -> str:
    """파일을 처리합니다.

    Args:
        file_path: 처리할 파일 경로

    Returns:
        처리 결과 또는 에러 메시지
    """
    import os

    # 파일 존재 확인
    if not os.path.exists(file_path):
        return f"Error: 파일을 찾을 수 없습니다: {file_path}"

    # 파일 타입 확인
    if not file_path.endswith((".txt", ".md", ".py")):
        return f"Error: 지원하지 않는 파일 형식입니다. 지원: .txt, .md, .py"

    try:
        with open(file_path, "r") as f:
            content = f.read()
        return f"파일 처리 완료. 길이: {len(content)}자"
    except Exception as e:
        return f"Error: 파일 읽기 실패: {str(e)}"
```

### 재시도 로직

```python
import time
from functools import wraps

def retry(max_attempts: int = 3, delay: float = 1.0):
    """재시도 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (attempt + 1))
            return f"Error after {max_attempts} attempts: {str(last_error)}"
        return wrapper
    return decorator

@tool
@retry(max_attempts=3, delay=2.0)
def reliable_api_call(endpoint: str) -> dict:
    """재시도 로직이 적용된 API 호출"""
    # API 호출 구현
    pass
```

---

## 모범 사례

### 1. 명확한 타입 힌트

```python
# ✅ 좋은 예
@tool
def search(query: str, limit: int = 10) -> list[dict]:
    """..."""

# ❌ 나쁜 예
@tool
def search(query, limit=10):
    """..."""
```

### 2. 안전한 기본값

```python
# ✅ 좋은 예: 안전한 기본값
@tool
def execute_query(query: str, timeout: int = 30) -> str:
    """타임아웃이 있는 쿼리 실행"""

# ❌ 나쁜 예: 위험한 기본값
@tool
def execute_query(query: str, timeout: int = 0) -> str:  # 무한 대기
    """..."""
```

### 3. 결과 크기 제한

```python
@tool
def search_large_dataset(query: str, max_results: int = 100) -> list[dict]:
    """대용량 데이터셋 검색

    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수 (최대 1000)

    Returns:
        검색 결과 (max_results로 제한)
    """
    # 최대값 제한
    max_results = min(max_results, 1000)

    results = perform_search(query)
    return results[:max_results]
```

### 4. 민감 정보 보호

```python
@tool
def call_api(endpoint: str) -> dict:
    """외부 API 호출

    Note:
        API 키는 환경 변수에서 로드됩니다.
        절대 API 키를 하드코딩하지 마세요.
    """
    import os
    api_key = os.environ.get("API_KEY")

    if not api_key:
        return {"error": "API_KEY 환경 변수를 설정하세요"}

    # API 호출 (키를 로그에 남기지 않음)
    pass
```

---

## 도구 테스트

### 단위 테스트

```python
import pytest

def test_calculate():
    """계산 도구 테스트"""
    assert calculate.invoke({"expression": "2 + 3"}) == 5
    assert calculate.invoke({"expression": "10 / 2"}) == 5.0

def test_calculate_error():
    """에러 처리 테스트"""
    result = calculate.invoke({"expression": "invalid"})
    assert "Error" in result
```

### 통합 테스트

```python
def test_tool_with_agent():
    """에이전트와 도구 통합 테스트"""
    agent = create_deep_agent(
        tools=[calculate],
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": "2 + 3을 계산해줘"}]
    })

    # 결과에 5가 포함되어야 함
    assert "5" in result["messages"][-1].content
```

---

## 다음 단계

- [Human-in-the-Loop 패턴](./human-in-the-loop.md)
- [병렬 서브에이전트 패턴](./parallel-subagents.md)
- [create_deep_agent API](../05-api-reference/create-deep-agent-api.md)
