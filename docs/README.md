# Deep Agents 문서

> **LangChain 기반 에이전트 하네스 프레임워크 - 기업 수준의 에이전트 구축을 위한 종합 가이드**

Deep Agents는 LangChain과 LangGraph를 기반으로 한 Python 에이전트 프레임워크입니다. 계획 수립(Planning), 파일 시스템 접근, 서브에이전트 위임 등의 고급 기능을 기본으로 제공하여 복잡한 작업을 수행하는 지능형 에이전트를 쉽게 구축할 수 있습니다.

## 빠른 시작

```python
from deepagents import create_deep_agent

# 기본 에이전트 생성 (Claude Sonnet 4.5 사용)
agent = create_deep_agent()

# 에이전트 실행
result = agent.invoke({"messages": [{"role": "user", "content": "프로젝트 구조를 분석해주세요"}]})
```

## 문서 구조

### [01. 아키텍처](./01-architecture/)
Deep Agents의 내부 구조와 설계 원리를 이해합니다.

| 문서 | 설명 |
|------|------|
| [시스템 개요](./01-architecture/overview.md) | 전체 시스템 아키텍처와 데이터 흐름 |
| [미들웨어 시스템](./01-architecture/middleware-system.md) | 미들웨어 스택의 동작 원리 |
| [백엔드 시스템](./01-architecture/backend-system.md) | 백엔드 추상화 패턴 |
| [상태 관리](./01-architecture/state-management.md) | LangGraph 상태와 데이터 흐름 |

### [02. 핵심 개념](./02-core-concepts/)
Deep Agents를 구성하는 핵심 개념들을 상세히 다룹니다.

| 문서 | 설명 |
|------|------|
| [create_deep_agent](./02-core-concepts/create-deep-agent.md) | 팩토리 함수 심층 분석 |
| [서브에이전트 오케스트레이션](./02-core-concepts/subagent-orchestration.md) | 작업 위임 패턴 |
| [스킬 시스템](./02-core-concepts/skills-system.md) | SKILL.md 포맷과 활용 |
| [메모리 시스템](./02-core-concepts/memory-system.md) | AGENTS.md 영속 컨텍스트 |

### [03. 에이전트 구축](./03-building-agents/)
실전 튜토리얼을 통해 다양한 유형의 에이전트를 구축합니다.

| 문서 | 설명 |
|------|------|
| [빠른 시작](./03-building-agents/quickstart.md) | 5분 만에 첫 에이전트 만들기 |
| [리서치 에이전트](./03-building-agents/research-agent-tutorial.md) | 웹 리서치 에이전트 구축 |
| [SQL 에이전트](./03-building-agents/sql-agent-tutorial.md) | 데이터베이스 쿼리 에이전트 |
| [콘텐츠 에이전트](./03-building-agents/content-agent-tutorial.md) | 콘텐츠 생성 에이전트 |

### [04. 패턴](./04-patterns/)
실무에서 자주 사용되는 디자인 패턴을 설명합니다.

| 문서 | 설명 |
|------|------|
| [병렬 서브에이전트](./04-patterns/parallel-subagents.md) | 동시 작업 실행 |
| [컨텍스트 관리](./04-patterns/context-management.md) | 토큰 관리와 요약 |
| [Human-in-the-Loop](./04-patterns/human-in-the-loop.md) | 승인 워크플로우 |
| [커스텀 도구](./04-patterns/custom-tools.md) | 도구 생성 패턴 |

### [05. API 레퍼런스](./05-api-reference/)
모든 API에 대한 상세 문서입니다.

| 문서 | 설명 |
|------|------|
| [create_deep_agent API](./05-api-reference/create-deep-agent-api.md) | 전체 파라미터 레퍼런스 |
| [미들웨어 API](./05-api-reference/middleware-api.md) | 모든 미들웨어 클래스 |
| [백엔드 API](./05-api-reference/backend-api.md) | 백엔드 프로토콜과 구현체 |
| [서브에이전트 API](./05-api-reference/subagent-api.md) | SubAgent/CompiledSubAgent 스펙 |

### [06. 고급 주제](./06-advanced/)
커스터마이징과 확장을 위한 고급 가이드입니다.

| 문서 | 설명 |
|------|------|
| [커스텀 미들웨어](./06-advanced/custom-middleware.md) | 미들웨어 작성법 |
| [커스텀 백엔드](./06-advanced/custom-backends.md) | BackendProtocol 구현 |
| [컴파일된 서브에이전트](./06-advanced/compiled-subagents.md) | 사전 빌드된 LangGraph 에이전트 |
| [프로덕션 배포](./06-advanced/production-deployment.md) | 엔터프라이즈 고려사항 |

---

## 핵심 특징

### 1. 기본 제공 도구들
Deep Agent는 생성 시 자동으로 다음 도구들을 포함합니다:

- **`write_todos`**: 할 일 목록 관리
- **`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`**: 파일 작업
- **`execute`**: 셸 명령 실행 (SandboxBackend 필요)
- **`task`**: 서브에이전트 호출

### 2. 미들웨어 스택
확장 가능한 미들웨어 시스템으로 에이전트 동작을 커스터마이징합니다:

```python
# 기본 미들웨어 스택
[
    TodoListMiddleware(),      # 할 일 관리
    MemoryMiddleware(),        # AGENTS.md 로딩
    SkillsMiddleware(),        # SKILL.md 로딩
    FilesystemMiddleware(),    # 파일 시스템 도구
    SubAgentMiddleware(),      # 서브에이전트 관리
    SummarizationMiddleware(), # 컨텍스트 요약
    AnthropicPromptCachingMiddleware(),  # 프롬프트 캐싱
    PatchToolCallsMiddleware(), # 도구 호출 패치
]
```

### 3. 서브에이전트 시스템
복잡한 작업을 독립적인 서브에이전트에 위임합니다:

```python
research_subagent = {
    "name": "researcher",
    "description": "웹 리서치를 수행하는 전문 에이전트",
    "system_prompt": "당신은 철저한 리서치를 수행하는 연구원입니다.",
    "tools": [web_search_tool],
}

agent = create_deep_agent(
    subagents=[research_subagent],
)
```

### 4. 백엔드 추상화
다양한 저장소와 실행 환경을 지원합니다:

- **StateBackend**: 에이전트 상태 내 임시 저장소 (기본값)
- **FilesystemBackend**: 로컬 파일 시스템
- **CompositeBackend**: 경로 기반 라우팅
- **SandboxBackendProtocol**: 명령 실행 지원

---

## 기술 스택

- **LangChain**: LLM 상호작용 및 에이전트 생성
- **LangGraph**: 그래프 기반 에이전트 오케스트레이션
- **Python 3.11+**: 타입 힌트 및 최신 문법 지원

## 지원 모델

기본적으로 Claude Sonnet 4.5를 사용하지만, `provider:model` 형식으로 다양한 모델을 사용할 수 있습니다:

```python
# OpenAI
agent = create_deep_agent(model="openai:gpt-4o")

# Google
agent = create_deep_agent(model="google:gemini-1.5-pro")

# Anthropic (기본값)
agent = create_deep_agent(model="anthropic:claude-sonnet-4-5-20250929")
```

---

## 다음 단계

1. **[빠른 시작](./03-building-agents/quickstart.md)**: 첫 번째 에이전트를 만들어 보세요
2. **[아키텍처 개요](./01-architecture/overview.md)**: 시스템 구조를 이해하세요
3. **[튜토리얼](./03-building-agents/)**: 실전 예제를 따라해 보세요

---

## 기여하기

이 문서에 대한 피드백이나 개선 사항이 있다면 Issue를 생성하거나 PR을 보내주세요.
