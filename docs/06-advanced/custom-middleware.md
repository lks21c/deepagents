# 커스텀 미들웨어 작성

> AgentMiddleware를 확장하여 사용자 정의 미들웨어를 작성하는 방법입니다.

## 개요

Deep Agents의 미들웨어 시스템은 에이전트의 모델 호출과 도구 호출을 가로채고 수정할 수 있는 확장 포인트를 제공합니다. 이를 통해 로깅, 캐싱, 검증, 변환 등 다양한 횡단 관심사(Cross-cutting Concerns)를 구현할 수 있습니다.

---

## AgentMiddleware 기본 클래스

**소스 위치**: `langchain.agents.middleware.types`

```python
from abc import ABC
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

class AgentMiddleware(ABC):
    """에이전트 미들웨어 기본 클래스"""

    state_schema: type[AgentState] | None = None
    """상태 스키마 확장 (선택)"""

    tools: list = []
    """미들웨어가 제공하는 도구 목록 (선택)"""

    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> StateUpdate | None:
        """에이전트 실행 전 호출 (선택)"""
        pass

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """모델 호출을 감싸서 수정 (선택)"""
        return handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """도구 호출을 감싸서 수정 (선택)"""
        return handler(request)
```

---

## 확장 포인트

### 1. state_schema - 상태 스키마 확장

미들웨어가 추가 상태 필드를 필요로 할 때 사용합니다.

```python
from typing import Annotated, NotRequired
from langchain.agents.middleware.types import AgentState, PrivateStateAttr

class MyMiddlewareState(AgentState):
    """커스텀 미들웨어 상태 스키마"""

    # 에이전트 상태에 추가되는 필드
    my_data: NotRequired[Annotated[dict[str, str], PrivateStateAttr]]
    """PrivateStateAttr: 서브에이전트 경계에서 필터링됨"""

class MyMiddleware(AgentMiddleware):
    state_schema = MyMiddlewareState

    def before_agent(self, state, runtime, config):
        # state["my_data"]에 접근 가능
        if "my_data" not in state:
            return {"my_data": {}}
        return None
```

**PrivateStateAttr 어노테이션**:
- 서브에이전트로 전달될 때 자동으로 제외
- 메인 에이전트와 서브에이전트 간 상태 격리
- `skills_metadata`, `memory_contents` 등이 이 어노테이션 사용

---

### 2. tools - 도구 제공

미들웨어가 에이전트에 도구를 추가할 수 있습니다.

```python
from langchain_core.tools import StructuredTool

class MyMiddleware(AgentMiddleware):
    def __init__(self):
        self.tools = [self._create_my_tool()]

    def _create_my_tool(self):
        def my_tool(param: str) -> str:
            """도구 설명"""
            return f"결과: {param}"

        return StructuredTool.from_function(
            name="my_tool",
            description="도구 설명",
            func=my_tool,
        )
```

---

### 3. before_agent - 에이전트 실행 전 훅

에이전트가 시작되기 전에 상태를 초기화하거나 수정합니다.

```python
class ContextLoaderMiddleware(AgentMiddleware):
    """외부 소스에서 컨텍스트를 로드하는 미들웨어"""

    def __init__(self, context_source: str):
        self.context_source = context_source

    def before_agent(self, state, runtime, config):
        # 컨텍스트 로드
        context = self._load_context(self.context_source)

        # 상태 업데이트 반환
        return {"loaded_context": context}

    def _load_context(self, source: str) -> dict:
        # 구현
        pass
```

**반환값**:
- `None`: 상태 변경 없음
- `dict`: 상태에 병합할 업데이트

---

### 4. wrap_model_call - 모델 호출 래핑

LLM 호출을 가로채서 요청/응답을 수정합니다.

```python
from langchain.agents.middleware.types import ModelRequest, ModelResponse

class LoggingMiddleware(AgentMiddleware):
    """모든 LLM 호출을 로깅하는 미들웨어"""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # 1. 요청 로깅
        print(f"LLM 요청: {len(request.messages)} 메시지")

        # 2. 요청 수정 (선택)
        modified_request = request.override(
            system_message=request.system_message + "\n[로깅 활성화]"
        )

        # 3. 핸들러 호출
        response = handler(modified_request)

        # 4. 응답 로깅
        print(f"LLM 응답: {response.message.content[:100]}...")

        return response
```

**ModelRequest 주요 속성**:
```python
class ModelRequest:
    messages: list[BaseMessage]      # 대화 메시지
    system_message: str              # 시스템 프롬프트
    tools: list[BaseTool]            # 사용 가능한 도구
    runtime: ToolRuntime             # 런타임 컨텍스트

    def override(self, **kwargs) -> ModelRequest:
        """속성을 오버라이드한 새 요청 반환"""
```

---

### 5. wrap_tool_call - 도구 호출 래핑

도구 실행을 가로채서 입력/출력을 수정합니다.

```python
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

class ToolValidationMiddleware(AgentMiddleware):
    """도구 호출을 검증하는 미들웨어"""

    def __init__(self, blocked_tools: list[str]):
        self.blocked_tools = blocked_tools

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        # 1. 차단된 도구 검사
        tool_name = request.tool_call["name"]
        if tool_name in self.blocked_tools:
            return ToolMessage(
                content=f"Error: {tool_name} 도구는 사용할 수 없습니다.",
                tool_call_id=request.tool_call["id"],
            )

        # 2. 핸들러 호출
        result = handler(request)

        # 3. 결과 후처리 (선택)
        if isinstance(result, ToolMessage):
            print(f"도구 {tool_name} 실행 완료")

        return result
```

**ToolCallRequest 주요 속성**:
```python
class ToolCallRequest:
    tool_call: dict[str, Any]  # {"name": "...", "args": {...}, "id": "..."}
    runtime: ToolRuntime       # 런타임 컨텍스트
```

---

## 비동기 메서드

모든 래핑 메서드는 비동기 버전을 제공합니다.

```python
class AsyncMiddleware(AgentMiddleware):
    """비동기 미들웨어 예시"""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        # 비동기 전처리
        await self._async_preprocess(request)

        # 핸들러 호출
        response = await handler(request)

        # 비동기 후처리
        await self._async_postprocess(response)

        return response

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        result = await handler(request)
        return result
```

---

## 실전 예시: 캐싱 미들웨어

```python
import hashlib
import json
from typing import Callable
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

class CachingMiddleware(AgentMiddleware):
    """LLM 응답을 캐싱하는 미들웨어

    동일한 입력에 대해 캐시된 응답을 반환하여
    API 호출 비용과 지연 시간을 줄입니다.
    """

    def __init__(self, cache_backend: dict | None = None):
        """캐싱 미들웨어 초기화

        Args:
            cache_backend: 캐시 저장소 (기본: 인메모리 딕셔너리)
        """
        self._cache = cache_backend if cache_backend is not None else {}

    def _compute_cache_key(self, request: ModelRequest) -> str:
        """요청에 대한 캐시 키 계산

        메시지 내용과 도구 이름을 기반으로 해시 생성
        """
        key_data = {
            "messages": [
                {"role": m.type, "content": str(m.content)}
                for m in request.messages
            ],
            "system": request.system_message,
            "tools": [t.name for t in request.tools],
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # 캐시 키 계산
        cache_key = self._compute_cache_key(request)

        # 캐시 히트 확인
        if cache_key in self._cache:
            print(f"캐시 히트: {cache_key[:8]}...")
            return self._cache[cache_key]

        # 캐시 미스: 실제 LLM 호출
        response = handler(request)

        # 응답 캐싱
        self._cache[cache_key] = response
        print(f"캐시 저장: {cache_key[:8]}...")

        return response
```

---

## 실전 예시: 속도 제한 미들웨어

```python
import time
from typing import Callable
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

class RateLimitMiddleware(AgentMiddleware):
    """API 호출 속도를 제한하는 미들웨어

    토큰 버킷 알고리즘을 사용하여 분당 최대 호출 수를 제한합니다.
    """

    def __init__(
        self,
        calls_per_minute: int = 60,
        retry_after: float = 1.0,
    ):
        """속도 제한 미들웨어 초기화

        Args:
            calls_per_minute: 분당 최대 호출 수
            retry_after: 제한 초과 시 대기 시간(초)
        """
        self._calls_per_minute = calls_per_minute
        self._retry_after = retry_after
        self._call_times: list[float] = []

    def _check_rate_limit(self) -> bool:
        """속도 제한 확인

        Returns:
            True: 호출 가능
            False: 제한 초과
        """
        now = time.time()
        minute_ago = now - 60.0

        # 1분 내의 호출만 유지
        self._call_times = [t for t in self._call_times if t > minute_ago]

        return len(self._call_times) < self._calls_per_minute

    def _record_call(self):
        """호출 시간 기록"""
        self._call_times.append(time.time())

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # 속도 제한 확인
        while not self._check_rate_limit():
            print(f"속도 제한 초과, {self._retry_after}초 대기...")
            time.sleep(self._retry_after)

        # 호출 기록 및 실행
        self._record_call()
        return handler(request)
```

---

## 실전 예시: 프롬프트 인젝션 방어 미들웨어

```python
import re
from typing import Callable
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)

class PromptInjectionDefenseMiddleware(AgentMiddleware):
    """프롬프트 인젝션 공격을 방어하는 미들웨어

    사용자 입력에서 의심스러운 패턴을 탐지하고 차단합니다.
    """

    # 의심스러운 패턴 목록
    SUSPICIOUS_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|above)\s+instructions",
        r"disregard\s+(all\s+)?(previous|above)\s+instructions",
        r"you\s+are\s+now\s+a",
        r"pretend\s+to\s+be",
        r"act\s+as\s+if",
        r"jailbreak",
        r"DAN\s+mode",
    ]

    def __init__(self, action: str = "block"):
        """프롬프트 인젝션 방어 미들웨어 초기화

        Args:
            action: 탐지 시 동작 ("block" | "warn" | "sanitize")
        """
        self._action = action
        self._patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.SUSPICIOUS_PATTERNS
        ]

    def _check_injection(self, text: str) -> bool:
        """프롬프트 인젝션 탐지

        Returns:
            True: 인젝션 의심
            False: 정상
        """
        for pattern in self._patterns:
            if pattern.search(text):
                return True
        return False

    def _sanitize(self, text: str) -> str:
        """의심스러운 패턴 제거"""
        for pattern in self._patterns:
            text = pattern.sub("[FILTERED]", text)
        return text

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # 최신 사용자 메시지 확인
        for message in reversed(request.messages):
            if message.type == "human":
                content = str(message.content)

                if self._check_injection(content):
                    if self._action == "block":
                        # 경고 메시지 반환
                        raise ValueError(
                            "프롬프트 인젝션 시도가 탐지되었습니다."
                        )
                    elif self._action == "warn":
                        print(f"경고: 의심스러운 입력 탐지: {content[:50]}...")
                    elif self._action == "sanitize":
                        # 메시지 정제
                        # (실제로는 메시지를 수정하는 로직 필요)
                        pass
                break

        return handler(request)
```

---

## 미들웨어 등록

### create_deep_agent에서 등록

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    middleware=[
        CachingMiddleware(),
        RateLimitMiddleware(calls_per_minute=30),
        LoggingMiddleware(),
    ],
)
```

### 미들웨어 실행 순서

미들웨어는 등록 순서대로 체인 형태로 실행됩니다:

```
요청 → MW1 → MW2 → MW3 → 모델/도구 → MW3 → MW2 → MW1 → 응답
```

**기본 스택 후 사용자 미들웨어 적용**:
1. TodoListMiddleware
2. MemoryMiddleware (선택)
3. SkillsMiddleware (선택)
4. FilesystemMiddleware
5. SubAgentMiddleware
6. SummarizationMiddleware
7. AnthropicPromptCachingMiddleware
8. PatchToolCallsMiddleware
9. **사용자 미들웨어** ← 여기에 추가됨
10. HumanInTheLoopMiddleware (선택)

---

## 시스템 프롬프트 수정 유틸리티

```python
from deepagents.middleware._utils import append_to_system_message

class MyMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        # 시스템 메시지에 지시사항 추가
        new_system = append_to_system_message(
            request.system_message,
            "## 추가 지시사항\n\n항상 친절하게 응답하세요."
        )

        modified_request = request.override(system_message=new_system)
        return handler(modified_request)
```

---

## 모범 사례

### 1. 단일 책임 원칙

```python
# ✅ 좋은 예: 하나의 관심사에 집중
class LoggingMiddleware(AgentMiddleware):
    """로깅만 담당"""

class CachingMiddleware(AgentMiddleware):
    """캐싱만 담당"""

# ❌ 나쁜 예: 여러 관심사 혼합
class DoEverythingMiddleware(AgentMiddleware):
    """로깅, 캐싱, 검증 모두 처리"""
```

### 2. 핸들러 항상 호출

```python
def wrap_model_call(self, request, handler):
    # ✅ 항상 핸들러를 호출하거나 명시적으로 처리
    try:
        return handler(request)
    except Exception as e:
        # 에러 처리
        raise

def wrap_model_call(self, request, handler):
    # ❌ 핸들러 호출 누락 - 체인이 끊어짐
    return None
```

### 3. 상태 불변성 유지

```python
def wrap_model_call(self, request, handler):
    # ✅ override()로 새 객체 생성
    modified = request.override(system_message="...")
    return handler(modified)

def wrap_model_call(self, request, handler):
    # ❌ 원본 객체 직접 수정
    request.system_message = "..."  # 위험!
    return handler(request)
```

### 4. 에러 처리

```python
def wrap_tool_call(self, request, handler):
    try:
        return handler(request)
    except Exception as e:
        # 에러를 ToolMessage로 변환
        return ToolMessage(
            content=f"Error: {e}",
            tool_call_id=request.tool_call["id"],
        )
```

---

## 테스트

```python
import pytest
from unittest.mock import Mock, MagicMock

def test_caching_middleware():
    """캐싱 미들웨어 테스트"""
    middleware = CachingMiddleware()

    # 모의 핸들러
    mock_handler = Mock(return_value=ModelResponse(...))

    # 첫 번째 호출: 캐시 미스
    request = ModelRequest(messages=[...], ...)
    response1 = middleware.wrap_model_call(request, mock_handler)
    assert mock_handler.call_count == 1

    # 두 번째 호출: 캐시 히트
    response2 = middleware.wrap_model_call(request, mock_handler)
    assert mock_handler.call_count == 1  # 핸들러 호출 안 함
    assert response1 == response2

def test_rate_limit_middleware():
    """속도 제한 미들웨어 테스트"""
    middleware = RateLimitMiddleware(calls_per_minute=2)

    mock_handler = Mock(return_value=ModelResponse(...))
    request = ModelRequest(messages=[...], ...)

    # 2번은 즉시 성공
    middleware.wrap_model_call(request, mock_handler)
    middleware.wrap_model_call(request, mock_handler)

    # 3번째는 대기 필요 (테스트에서는 시간 모킹)
    # ...
```

---

## 관련 문서

- [미들웨어 시스템](../01-architecture/middleware-system.md)
- [미들웨어 API](../05-api-reference/middleware-api.md)
- [create_deep_agent API](../05-api-reference/create-deep-agent-api.md)
