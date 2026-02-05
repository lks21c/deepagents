# 메모리 시스템 (Memory System)

> AGENTS.md 파일을 통한 에이전트의 영속적 컨텍스트와 학습 메커니즘을 설명합니다.

## 개요

메모리 시스템은 에이전트가 세션 간에 컨텍스트를 유지하고, 사용자로부터 학습한 내용을 저장하는 메커니즘입니다. [AGENTS.md 명세](https://agents.md/)를 구현합니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Memory System Flow                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │                    Memory Sources                           │    │
│   │                                                             │    │
│   │   ~/.deepagents/AGENTS.md    (글로벌 사용자 설정)           │    │
│   │           │                                                 │    │
│   │           ▼                                                 │    │
│   │   ./project/AGENTS.md        (프로젝트별 설정)              │    │
│   │           │                                                 │    │
│   │           ▼                                                 │    │
│   │   ./team/AGENTS.md           (팀별 설정)                    │    │
│   │                                                             │    │
│   └────────────────────────────────────────────────────────────┘    │
│                               │                                      │
│                               ▼                                      │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │                   MemoryMiddleware                          │    │
│   │                                                             │    │
│   │   1. before_agent: 소스에서 메모리 로드                     │    │
│   │   2. wrap_model_call: 시스템 프롬프트에 주입                │    │
│   │                                                             │    │
│   └────────────────────────────────────────────────────────────┘    │
│                               │                                      │
│                               ▼                                      │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │                    System Prompt                            │    │
│   │                                                             │    │
│   │   <agent_memory>                                            │    │
│   │   ~/.deepagents/AGENTS.md                                   │    │
│   │   [글로벌 메모리 내용]                                      │    │
│   │                                                             │    │
│   │   ./project/AGENTS.md                                       │    │
│   │   [프로젝트 메모리 내용]                                    │    │
│   │   </agent_memory>                                           │    │
│   │                                                             │    │
│   │   <memory_guidelines>                                       │    │
│   │   [학습 및 업데이트 지침]                                   │    │
│   │   </memory_guidelines>                                      │    │
│   │                                                             │    │
│   └────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## MemoryMiddleware

**소스 위치**: `libs/deepagents/deepagents/middleware/memory.py:167-421`

### 클래스 정의

```python
class MemoryMiddleware(AgentMiddleware):
    """AGENTS.md 파일에서 에이전트 메모리를 로드하는 미들웨어

    Args:
        backend: 파일 작업을 위한 백엔드 인스턴스 또는 팩토리 함수
        sources: 메모리 파일 경로 목록
    """

    state_schema = MemoryState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES,
        sources: list[str],
    ) -> None:
        self._backend = backend
        self.sources = sources
```

**파라미터 설명**:

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `backend` | `BACKEND_TYPES` | 파일 읽기를 위한 백엔드 |
| `sources` | `list[str]` | AGENTS.md 파일 경로 목록 |

---

## 메모리 로드 과정

### 1. before_agent 훅

```python
def before_agent(
    self,
    state: MemoryState,
    runtime: Runtime,
    config: RunnableConfig,
) -> MemoryStateUpdate | None:
    """에이전트 실행 전 메모리 로드"""

    # 이미 로드되었으면 건너뛰기
    if "memory_contents" in state:
        return None

    backend = self._get_backend(state, runtime, config)
    contents: dict[str, str] = {}

    for path in self.sources:
        content = self._load_memory_from_backend_sync(backend, path)
        if content:
            contents[path] = content
            logger.debug(f"Loaded memory from: {path}")

    return MemoryStateUpdate(memory_contents=contents)
```

**설계 이유**:
- **중복 로드 방지**: `memory_contents`가 이미 상태에 있으면 건너뜀
- **점진적 실패**: 파일이 없어도 계속 진행 (graceful degradation)
- **순서 유지**: `sources` 순서대로 로드

### 2. wrap_model_call 훅

```python
def wrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """시스템 프롬프트에 메모리 주입"""
    modified_request = self.modify_request(request)
    return handler(modified_request)

def modify_request(self, request: ModelRequest) -> ModelRequest:
    """메모리 내용을 시스템 메시지에 추가"""
    contents = request.state.get("memory_contents", {})
    agent_memory = self._format_agent_memory(contents)

    new_system_message = append_to_system_message(
        request.system_message,
        agent_memory
    )

    return request.override(system_message=new_system_message)
```

---

## 메모리 포맷

### 시스템 프롬프트 템플릿

**소스 위치**: `memory.py:105-164`

```python
MEMORY_SYSTEM_PROMPT = """<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
    The above <agent_memory> was loaded in from files in your filesystem.
    As you learn from your interactions with the user, you can save new
    knowledge by calling the `edit_file` tool.

    **Learning from feedback:**
    - One of your MAIN PRIORITIES is to learn from your interactions.
    - When you need to remember something, updating memory must be your
      FIRST, IMMEDIATE action.
    - When user says something is better/worse, capture WHY and encode
      it as a pattern.
    ...
</memory_guidelines>
"""
```

**템플릿 구성요소**:

| 섹션 | 내용 |
|------|------|
| `<agent_memory>` | 로드된 AGENTS.md 내용 |
| `<memory_guidelines>` | 메모리 업데이트 지침 |

### 포맷 함수

```python
def _format_agent_memory(self, contents: dict[str, str]) -> str:
    """메모리를 위치와 내용 쌍으로 포맷"""
    if not contents:
        return MEMORY_SYSTEM_PROMPT.format(
            agent_memory="(No memory loaded)"
        )

    sections = []
    for path in self.sources:
        if contents.get(path):
            sections.append(f"{path}\n{contents[path]}")

    memory_body = "\n\n".join(sections)
    return MEMORY_SYSTEM_PROMPT.format(agent_memory=memory_body)
```

**출력 예시**:

```markdown
<agent_memory>
~/.deepagents/AGENTS.md
# 글로벌 설정
- 한국어로 응답
- 코드에 타입 힌트 사용

./project/AGENTS.md
# 프로젝트 정보
- FastAPI 기반 백엔드
- PostgreSQL 데이터베이스
</agent_memory>
```

---

## AGENTS.md 파일 형식

### 기본 구조

```markdown
# Agent Memory

## 프로젝트 개요
이 프로젝트는 전자상거래 플랫폼입니다.

## 빌드/테스트 명령
- 빌드: `npm run build`
- 테스트: `npm test`
- 린트: `npm run lint`

## 코드 스타일 가이드라인
- TypeScript strict 모드 사용
- 함수형 컴포넌트 선호
- 모든 API 응답에 타입 정의

## 아키텍처 노트
- 모노레포 구조 (Turborepo)
- 마이크로서비스 백엔드

## 사용자 선호
(사용자 피드백에 따라 업데이트됨)
- 응답은 한국어로
- 코드 예제에 주석 포함
```

**권장 섹션**:

| 섹션 | 용도 |
|------|------|
| 프로젝트 개요 | 프로젝트의 목적과 범위 |
| 빌드/테스트 명령 | 개발 환경 설정 |
| 코드 스타일 | 코딩 컨벤션 |
| 아키텍처 노트 | 시스템 구조 |
| 사용자 선호 | 학습된 사용자 선호도 |

---

## 메모리 학습 메커니즘

### 학습 트리거

**소스 위치**: `memory.py:113-140`

메모리 가이드라인에 정의된 학습 시점:

```markdown
**When to update memories:**
- 사용자가 명시적으로 기억을 요청할 때
  예: "내 이메일 기억해", "이 선호도 저장해"

- 사용자가 역할이나 행동 방식을 설명할 때
  예: "너는 웹 리서처야", "항상 X를 해"

- 사용자가 작업에 피드백을 줄 때
  → 무엇이 잘못되었고 어떻게 개선할지 캡처

- 도구 사용에 필요한 정보를 제공할 때
  예: Slack 채널 ID, 이메일 주소

- 향후 작업에 유용한 컨텍스트를 제공할 때
  예: 도구 사용법, 특정 상황에서의 행동

- 새로운 패턴이나 선호도를 발견할 때
  예: 코딩 스타일, 컨벤션, 워크플로우
```

### 학습하지 않을 때

```markdown
**When to NOT update memories:**
- 일시적/임시 정보
  예: "지금 늦어", "지금 폰으로 보는 중"

- 일회성 작업 요청
  예: "레시피 찾아줘", "25 * 4는?"

- 지속적 선호를 드러내지 않는 단순 질문
  예: "오늘 무슨 요일?", "X를 설명해줘"

- 인사말이나 가벼운 대화
  예: "좋아!", "안녕", "고마워"

- 향후 대화에서 오래되거나 무관한 정보

- API 키, 토큰, 비밀번호, 기타 자격증명은 절대 저장 금지
```

### 학습 예시

**예시 1: 사용자 정보 기억**

```
User: 내 구글 계정에 연결해줘
Agent: 구글 계정 이메일이 뭐예요?
User: john@example.com
Agent: 메모리에 저장하겠습니다.
Tool Call: edit_file(...) -> 사용자 구글 이메일이 john@example.com임을 기억
```

**예시 2: 암묵적 선호도 기억**

```
User: LangChain으로 딥 에이전트 만드는 예제 보여줘
Agent: 네, Python 예제입니다 <Python 코드>
User: JavaScript로 해줘
Agent: 메모리에 저장하겠습니다.
Tool Call: edit_file(...) -> 사용자가 LangChain 코드 예제를 JavaScript로 선호함
Agent: 네, JavaScript 예제입니다 <JavaScript 코드>
```

---

## 메모리 상태 스키마

### MemoryState

**소스 위치**: `memory.py:88-97`

```python
class MemoryState(AgentState):
    """MemoryMiddleware의 상태 스키마

    Attributes:
        memory_contents: 소스 경로 → 로드된 내용 매핑
            PrivateStateAttr로 표시되어 최종 에이전트 상태에 포함되지 않음
    """

    memory_contents: NotRequired[Annotated[dict[str, str], PrivateStateAttr]]
```

**`PrivateStateAttr` 사용 이유**:
- 메모리 내용은 에이전트 내부용
- 부모 에이전트에게 전파되지 않음
- 상태 직렬화 크기 최소화

---

## 사용 예제

### 기본 사용

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    memory=[
        "~/.deepagents/AGENTS.md",   # 글로벌 설정
        "./AGENTS.md",                # 프로젝트 설정
    ],
    backend=FilesystemBackend(root_dir="/"),
)
```

### 계층적 메모리 구조

```python
# 메모리 소스 계층 (나중 것이 우선)
agent = create_deep_agent(
    memory=[
        "/base/AGENTS.md",           # 기본 설정 (가장 낮은 우선순위)
        "/team/AGENTS.md",           # 팀 설정
        "/project/AGENTS.md",        # 프로젝트 설정
        "./user/AGENTS.md",          # 사용자 설정 (가장 높은 우선순위)
    ],
)
```

**계층 설계 이유**:
- 기본 설정 → 팀 설정 → 프로젝트 설정 → 사용자 설정
- 더 구체적인 설정이 일반적인 설정을 오버라이드

### 직접 미들웨어 사용

```python
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.backends import FilesystemBackend

memory_middleware = MemoryMiddleware(
    backend=FilesystemBackend(root_dir="/"),
    sources=[
        "~/.deepagents/AGENTS.md",
        "./project/AGENTS.md",
    ],
)

agent = create_deep_agent(
    middleware=[memory_middleware],
)
```

---

## 메모리 업데이트

에이전트는 `edit_file` 도구를 사용하여 메모리를 업데이트합니다.

```python
# 에이전트가 내부적으로 호출
edit_file(
    file_path="./AGENTS.md",
    old_string="## 사용자 선호\n(사용자 피드백에 따라 업데이트됨)",
    new_string="## 사용자 선호\n(사용자 피드백에 따라 업데이트됨)\n- 응답은 한국어로",
)
```

**주의사항**:
- 에이전트가 자동으로 메모리를 업데이트
- 사용자가 명시적으로 요청하지 않아도 학습 발생
- 민감한 정보(API 키, 비밀번호)는 저장 금지

---

## 문제 해결

### 메모리가 로드되지 않음

```python
# 경로 확인
import os
print(os.path.exists("./AGENTS.md"))

# 백엔드 root_dir 확인
backend = FilesystemBackend(root_dir="/correct/path")
```

### 메모리 업데이트가 안 됨

```python
# 파일 쓰기 권한 확인
import os
print(os.access("./AGENTS.md", os.W_OK))
```

### 민감한 정보 저장 방지

AGENTS.md에 다음을 추가:

```markdown
## 보안 규칙
- API 키, 토큰, 비밀번호는 절대 이 파일에 저장하지 않음
- 민감한 정보는 환경 변수 사용
```

---

## 다음 단계

- [스킬 시스템](./skills-system.md)
- [서브에이전트 오케스트레이션](./subagent-orchestration.md)
- [컨텍스트 관리 패턴](../04-patterns/context-management.md)
