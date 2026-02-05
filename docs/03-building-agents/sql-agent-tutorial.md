# SQL 에이전트 튜토리얼

> 자연어를 SQL 쿼리로 변환하여 데이터베이스를 조회하는 에이전트를 구축합니다.

## 개요

이 튜토리얼에서는 다음 기능을 가진 SQL 에이전트를 구축합니다:

- 자연어 질문을 SQL로 변환
- 스키마 탐색 및 이해
- 쿼리 실행 및 결과 해석
- 메모리와 스킬 시스템 활용

**소스 참조**: `examples/text-to-sql-agent/agent.py`

## 사전 요구사항

```bash
pip install deepagents langchain-community
```

## 프로젝트 구조

```
sql_agent/
├── agent.py           # 메인 에이전트 정의
├── chinook.db         # 샘플 SQLite 데이터베이스
├── AGENTS.md          # 에이전트 메모리/지침
└── skills/
    └── query-writing/
        └── SKILL.md   # SQL 작성 스킬
```

---

## Step 1: 데이터베이스 준비

Chinook 샘플 데이터베이스를 사용합니다 (음악 스토어 데이터).

```bash
# Chinook DB 다운로드
curl -L https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite -o chinook.db
```

---

## Step 2: AGENTS.md 작성

에이전트의 영속 메모리로 사용될 파일입니다.

### AGENTS.md

```markdown
# SQL Agent Identity

당신은 데이터베이스 전문가입니다. 자연어 질문을 SQL로 변환하고 결과를 해석합니다.

## 데이터베이스 정보

### Chinook Database
음악 스토어 데이터베이스입니다.

**주요 테이블:**
- `Artist`: 아티스트 정보
- `Album`: 앨범 정보 (ArtistId 외래키)
- `Track`: 트랙 정보 (AlbumId, GenreId, MediaTypeId 외래키)
- `Customer`: 고객 정보
- `Invoice`: 주문 정보 (CustomerId 외래키)
- `InvoiceLine`: 주문 상세 (InvoiceId, TrackId 외래키)
- `Employee`: 직원 정보

### 주의사항
- SELECT 쿼리만 실행 (데이터 수정 금지)
- 대량 결과는 LIMIT 사용
- 복잡한 쿼리는 단계별로 구성

## 사용자 선호

(여기에 학습된 내용이 추가됩니다)
```

---

## Step 3: 스킬 작성

SQL 쿼리 작성을 위한 스킬을 정의합니다.

### skills/query-writing/SKILL.md

```markdown
---
name: query-writing
description: SQL 쿼리 작성 및 최적화 워크플로우
---

# SQL Query Writing Skill

## When to Use
- 사용자가 데이터 조회를 요청할 때
- 복잡한 분석 쿼리가 필요할 때

## Workflow

### 1. 스키마 확인
```
sql_db_list_tables()  # 테이블 목록
sql_db_schema(table_names="Album,Artist")  # 스키마 확인
```

### 2. 쿼리 작성 원칙
- 필요한 컬럼만 SELECT
- 적절한 JOIN 사용
- WHERE 조건 명확히
- 대량 결과는 LIMIT

### 3. 쿼리 검증
- 먼저 작은 LIMIT으로 테스트
- 결과 확인 후 전체 실행

### 4. 결과 해석
- 데이터를 이해하기 쉽게 설명
- 필요시 추가 분석 제안

## Query Patterns

### 집계 쿼리
```sql
SELECT ArtistId, COUNT(*) as AlbumCount
FROM Album
GROUP BY ArtistId
ORDER BY AlbumCount DESC
LIMIT 10;
```

### JOIN 쿼리
```sql
SELECT a.Name as Artist, al.Title as Album
FROM Artist a
JOIN Album al ON a.ArtistId = al.ArtistId
WHERE a.Name LIKE '%Beatles%';
```
```

---

## Step 4: 에이전트 정의

### agent.py

```python
"""SQL Agent - Text-to-SQL Deep Agent"""

import os
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_anthropic import ChatAnthropic
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase


def create_sql_agent():
    """SQL Deep Agent를 생성합니다."""

    # 현재 디렉토리 기준 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # === 1. 데이터베이스 연결 ===
    db_path = os.path.join(base_dir, "chinook.db")
    db = SQLDatabase.from_uri(
        f"sqlite:///{db_path}",
        sample_rows_in_table_info=3,  # 스키마에 샘플 데이터 포함
    )

    # === 2. 모델 설정 ===
    model = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0,  # SQL 쿼리 정확도를 위해 0
    )

    # === 3. SQL 도구 생성 ===
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    sql_tools = toolkit.get_tools()

    # 제공되는 도구:
    # - sql_db_query: SQL 쿼리 실행
    # - sql_db_schema: 테이블 스키마 조회
    # - sql_db_list_tables: 테이블 목록 조회
    # - sql_db_query_checker: 쿼리 문법 검사

    # === 4. 에이전트 생성 ===
    agent = create_deep_agent(
        model=model,
        tools=sql_tools,
        memory=["./AGENTS.md"],  # 에이전트 메모리 로드
        skills=["./skills/"],    # 스킬 로드
        subagents=[],            # 서브에이전트 없음 (단순 작업)
        backend=FilesystemBackend(root_dir=base_dir),  # 파일 접근용
    )

    return agent


def query(question: str) -> str:
    """자연어 질문으로 데이터베이스를 조회합니다.

    Args:
        question: 자연어 질문

    Returns:
        쿼리 결과 및 해석
    """
    agent = create_sql_agent()
    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    return result["messages"][-1].content


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "가장 많이 팔린 아티스트 5명은 누구인가요?"

    print(f"📊 질문: {question}\n")
    print("-" * 50)
    print(query(question))
```

**코드 상세 설명**:

```python
db = SQLDatabase.from_uri(
    f"sqlite:///{db_path}",
    sample_rows_in_table_info=3,
)
```
- **sample_rows_in_table_info=3**: 스키마 조회 시 샘플 데이터 3행 포함
- LLM이 데이터 구조를 더 잘 이해할 수 있음

```python
toolkit = SQLDatabaseToolkit(db=db, llm=model)
sql_tools = toolkit.get_tools()
```
- LangChain의 SQL 툴킷 사용
- 자동으로 쿼리 실행, 스키마 조회, 검증 도구 생성

```python
memory=["./AGENTS.md"],
skills=["./skills/"],
```
- **memory**: 에이전트 역할과 데이터베이스 정보
- **skills**: SQL 작성 워크플로우

---

## Step 5: 실행

```bash
# 기본 질문
python agent.py

# 커스텀 질문
python agent.py "2010년에 가장 많이 판매된 장르는?"
python agent.py "직원별 담당 고객 수는?"
python agent.py "캐나다 고객들의 총 구매액은?"
```

### 예상 출력

```
📊 질문: 가장 많이 팔린 아티스트 5명은 누구인가요?

--------------------------------------------------

먼저 데이터베이스 스키마를 확인하겠습니다.

[sql_db_list_tables 호출]
[sql_db_schema 호출]

판매량을 계산하려면 InvoiceLine과 Track, Album, Artist를 조인해야 합니다.

```sql
SELECT
    a.Name as ArtistName,
    SUM(il.Quantity) as TotalSold
FROM Artist a
JOIN Album al ON a.ArtistId = al.ArtistId
JOIN Track t ON al.AlbumId = t.AlbumId
JOIN InvoiceLine il ON t.TrackId = il.TrackId
GROUP BY a.ArtistId
ORDER BY TotalSold DESC
LIMIT 5;
```

## 결과

| 순위 | 아티스트 | 총 판매량 |
|------|---------|----------|
| 1 | Iron Maiden | 140 |
| 2 | U2 | 107 |
| 3 | Metallica | 91 |
| 4 | Led Zeppelin | 87 |
| 5 | Os Paralamas Do Sucesso | 45 |

**분석**: Iron Maiden이 140건으로 가장 많이 팔렸으며, 상위 5명의 아티스트가
전체 판매의 상당 부분을 차지합니다.
```

---

## 고급 기능

### Human-in-the-Loop (쿼리 승인)

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_deep_agent(
    model=model,
    tools=sql_tools,
    memory=["./AGENTS.md"],
    skills=["./skills/"],
    backend=FilesystemBackend(root_dir=base_dir),
    checkpointer=MemorySaver(),  # 필수
    interrupt_on={
        "sql_db_query": True,  # 모든 쿼리 실행 전 승인
    },
)
```

### 결과 캐싱

```python
from langgraph.cache import InMemoryCache

agent = create_deep_agent(
    ...
    cache=InMemoryCache(),
)
```

---

## 문제 해결

### 스키마를 찾지 못함

```python
# 명시적으로 테이블 지정
db = SQLDatabase.from_uri(
    db_uri,
    include_tables=["Artist", "Album", "Track"],
)
```

### 쿼리 타임아웃

```python
db = SQLDatabase.from_uri(
    db_uri,
    max_string_length=300,  # 긴 텍스트 자르기
)
```

---

## 다음 단계

- [콘텐츠 에이전트 튜토리얼](./content-agent-tutorial.md)
- [메모리 시스템 상세](../02-core-concepts/memory-system.md)
- [스킬 시스템 상세](../02-core-concepts/skills-system.md)
