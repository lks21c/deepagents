# 빠른 시작 가이드

> 5분 만에 첫 번째 Deep Agent를 만들고 실행해봅니다.

## 사전 요구사항

- Python 3.11 이상
- Anthropic API 키 (Claude 모델 사용 시)

## 설치

```bash
pip install deepagents
```

## Step 1: 기본 에이전트 생성

가장 간단한 형태의 Deep Agent를 만들어봅니다.

```python
from deepagents import create_deep_agent

# 기본 설정으로 에이전트 생성
# - 모델: Claude Sonnet 4.5
# - 도구: 파일 시스템 (ls, read, write, edit, glob, grep), 할 일 관리
# - 서브에이전트: general-purpose
agent = create_deep_agent()

# 에이전트 실행
result = agent.invoke({
    "messages": [{"role": "user", "content": "안녕하세요!"}]
})

# 응답 출력
print(result["messages"][-1].content)
```

## Step 2: 환경 변수 설정

Anthropic API 키를 설정합니다.

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

또는 Python 코드에서:

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
```

## Step 3: 커스텀 도구 추가

웹 검색 도구를 추가해봅니다.

```python
from deepagents import create_deep_agent
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """웹에서 정보를 검색합니다.

    Args:
        query: 검색할 쿼리
    """
    # 실제 구현에서는 검색 API 호출
    return f"'{query}'에 대한 검색 결과: [검색 결과 데이터]"

# 커스텀 도구가 포함된 에이전트 생성
agent = create_deep_agent(
    tools=[web_search],
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "최신 Python 버전에 대해 검색해주세요"}]
})

print(result["messages"][-1].content)
```

## Step 4: 시스템 프롬프트 커스터마이징

에이전트의 역할을 정의합니다.

```python
agent = create_deep_agent(
    system_prompt="""당신은 Python 전문가입니다.

## 역할
- Python 코드 작성 및 리뷰
- 모범 사례와 PEP 스타일 가이드 준수
- 명확하고 간결한 설명 제공

## 규칙
- 항상 타입 힌트 사용
- 함수에는 docstring 작성
- 복잡한 로직에는 주석 추가
""",
    tools=[web_search],
)
```

## Step 5: 서브에이전트 추가

리서치 작업을 전담하는 서브에이전트를 추가합니다.

```python
from deepagents import create_deep_agent
from langchain_core.tools import tool

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """웹에서 정보를 검색합니다."""
    # Tavily 등 실제 검색 API 사용
    return f"검색 결과: {query}"

# 리서치 서브에이전트 정의
researcher = {
    "name": "researcher",
    "description": "웹 리서치를 수행하는 전문 에이전트. 복잡한 주제에 대한 정보 수집 시 사용.",
    "system_prompt": """당신은 철저한 리서치를 수행하는 연구원입니다.

## 작업 방식
1. 검색 전략 수립
2. 다양한 쿼리로 정보 수집
3. 수집한 정보 검증 및 정리
4. 구조화된 결과 반환

## 출력 형식
- 핵심 발견사항 요약
- 상세 내용 (출처 포함)
""",
    "tools": [web_search],
}

# 서브에이전트 포함 에이전트 생성
agent = create_deep_agent(
    system_prompt="당신은 기술 분석가입니다. 복잡한 리서치는 researcher에게 위임하세요.",
    subagents=[researcher],
    tools=[web_search],
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "2024년 AI 트렌드에 대해 조사해주세요"}]
})

print(result["messages"][-1].content)
```

## Step 6: 파일 시스템 작업

에이전트가 파일을 읽고 쓰도록 합니다.

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# 로컬 파일 시스템 백엔드 사용
agent = create_deep_agent(
    backend=FilesystemBackend(root_dir="./workspace"),
    system_prompt="당신은 코드 분석 도우미입니다.",
)

# 파일 분석 요청
result = agent.invoke({
    "messages": [{"role": "user", "content": "workspace 폴더의 Python 파일들을 분석해주세요"}]
})

# 에이전트가 사용할 수 있는 파일 도구:
# - ls("/workspace"): 디렉토리 목록
# - read_file("/workspace/main.py"): 파일 읽기
# - write_file("/workspace/new.py", content): 새 파일 생성
# - edit_file("/workspace/main.py", old_str, new_str): 파일 편집
# - glob("**/*.py"): 패턴 매칭
# - grep("def ", path="/workspace"): 텍스트 검색
```

## Step 7: 대화 지속성 (Checkpointer)

대화 히스토리를 유지합니다.

```python
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

# 체크포인터 설정
checkpointer = MemorySaver()

agent = create_deep_agent(
    checkpointer=checkpointer,
)

# 첫 번째 대화
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "제 이름은 김철수입니다"}]},
    config={"configurable": {"thread_id": "session-1"}}
)

# 같은 세션에서 계속
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "제 이름이 뭐였죠?"}]},
    config={"configurable": {"thread_id": "session-1"}}  # 같은 thread_id
)

print(result2["messages"][-1].content)
# 출력: "김철수님이라고 말씀하셨습니다."
```

## Step 8: 스트리밍 출력

실시간으로 응답을 받습니다.

```python
import asyncio
from deepagents import create_deep_agent

agent = create_deep_agent()

async def stream_response():
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "피보나치 함수를 설명해주세요"}]},
        stream_mode="values",
    ):
        if "messages" in chunk:
            latest = chunk["messages"][-1]
            if hasattr(latest, "content") and latest.content:
                print(latest.content, end="", flush=True)

asyncio.run(stream_response())
```

## 완전한 예제

모든 요소를 결합한 완전한 에이전트:

```python
import asyncio
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

# 커스텀 도구
@tool
def calculate(expression: str) -> float:
    """수학 표현식을 계산합니다.

    Args:
        expression: 계산할 수식 (예: "2 + 3 * 4")
    """
    return eval(expression)

# 서브에이전트
math_expert = {
    "name": "math-expert",
    "description": "복잡한 수학 문제를 해결하는 에이전트",
    "system_prompt": "당신은 수학 전문가입니다. 단계별로 풀이를 설명하세요.",
    "tools": [calculate],
}

# 에이전트 생성
agent = create_deep_agent(
    system_prompt="""당신은 만능 AI 어시스턴트입니다.

## 역할
- 코드 작성 및 분석
- 파일 관리
- 수학 문제는 math-expert에게 위임

## 응답 스타일
- 친절하고 명확하게
- 필요시 예제 코드 제공
""",
    tools=[calculate],
    subagents=[math_expert],
    backend=FilesystemBackend(root_dir="./workspace"),
    checkpointer=MemorySaver(),
)

# 실행
async def main():
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "workspace의 Python 파일을 분석하고 개선점을 제안해주세요"}]},
        config={"configurable": {"thread_id": "analysis-session"}},
        stream_mode="values",
    ):
        if "messages" in chunk:
            latest = chunk["messages"][-1]
            if hasattr(latest, "content") and latest.content:
                print(latest.content)

asyncio.run(main())
```

## 다음 단계

- [리서치 에이전트 튜토리얼](./research-agent-tutorial.md): 웹 리서치 에이전트 구축
- [SQL 에이전트 튜토리얼](./sql-agent-tutorial.md): 데이터베이스 쿼리 에이전트
- [콘텐츠 에이전트 튜토리얼](./content-agent-tutorial.md): 콘텐츠 생성 에이전트

## 문제 해결

### API 키 오류

```
AuthenticationError: Invalid API Key
```

→ `ANTHROPIC_API_KEY` 환경 변수 확인

### 모델 미지원 오류

```
Error: Tool calling not supported
```

→ 도구 호출을 지원하는 모델 사용 (Claude 3+, GPT-4+)

### 파일 접근 오류

```
Error: Path traversal not allowed
```

→ 절대 경로가 `root_dir` 내에 있는지 확인
