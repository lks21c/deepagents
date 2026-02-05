"""
모듈명: graph.py
설명: Deep Agent 생성 및 구성을 담당하는 핵심 모듈

이 모듈은 LangChain과 LangGraph를 기반으로 한 지능형 에이전트를 생성합니다.
Deep Agent는 계획 수립(planning), 파일 시스템 접근, 서브에이전트 관리 등의
고급 기능을 기본으로 제공합니다.

주요 기능:
    - create_deep_agent(): 완전한 기능을 갖춘 Deep Agent 인스턴스 생성
    - get_default_model(): 기본 LLM 모델(Claude Sonnet 4.5) 설정

의존성:
    - langchain: LLM 상호작용 및 에이전트 생성 프레임워크
    - langchain_anthropic: Anthropic Claude 모델 통합
    - langgraph: 그래프 기반 에이전트 오케스트레이션
    - deepagents.backends: 파일 저장소 및 실행 백엔드
    - deepagents.middleware: 다양한 에이전트 미들웨어 컴포넌트

사용 예시:
    >>> from deepagents import create_deep_agent
    >>> agent = create_deep_agent()  # 기본 설정으로 생성
    >>> agent = create_deep_agent(model="openai:gpt-4")  # 특정 모델 지정
"""

from collections.abc import Callable, Sequence
from typing import Any

# LangChain 핵심 컴포넌트 임포트
# - create_agent: 기본 에이전트 생성 함수
# - 미들웨어: 에이전트 동작을 확장하는 컴포넌트들
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain.chat_models import init_chat_model

# Anthropic Claude 모델 관련 임포트
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

# LangChain 핵심 타입 및 인터페이스
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool

# LangGraph 컴포넌트: 상태 관리 및 캐싱
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

# Deep Agents 내부 모듈
# - backends: 파일 저장소 및 샌드박스 실행 환경
# - middleware: 에이전트 기능 확장 미들웨어
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    CompiledSubAgent,
    SubAgent,
    SubAgentMiddleware,
)
from deepagents.middleware.summarization import SummarizationMiddleware, _compute_summarization_defaults

# 기본 에이전트 시스템 프롬프트
# 사용자 정의 프롬프트와 결합되어 에이전트의 기본 동작을 정의
BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."


def get_default_model() -> ChatAnthropic:
    """
    Deep Agent의 기본 LLM 모델을 반환합니다.

    이 함수는 Anthropic의 Claude Sonnet 4.5 모델을 기본값으로 설정합니다.
    최대 토큰 수는 20,000으로 설정되어 긴 컨텍스트 처리가 가능합니다.

    Returns:
        ChatAnthropic: Claude Sonnet 4.5로 구성된 ChatAnthropic 인스턴스

    Note:
        - 모델명: claude-sonnet-4-5-20250929
        - 최대 토큰: 20,000
        - 다른 모델을 사용하려면 create_deep_agent()의 model 파라미터를 사용하세요.
    """
    return ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=20000,  # type: ignore[call-arg]
    )


def create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph:
    """
    Deep Agent를 생성합니다.

    이 함수는 계획 수립, 파일 시스템 접근, 서브에이전트 관리 등의 고급 기능을
    갖춘 완전한 에이전트를 생성합니다. LangGraph의 CompiledStateGraph로
    반환되어 상태 관리와 복잡한 워크플로우를 지원합니다.

    !!! warning "Deep Agent는 도구 호출(tool calling)을 지원하는 LLM이 필요합니다!"

    기본 제공 도구:
        - write_todos: 할 일 목록 관리
        - ls, read_file, write_file, edit_file, glob, grep: 파일 작업
        - execute: 셸 명령어 실행 (SandboxBackendProtocol 구현 백엔드 필요)
        - task: 서브에이전트 호출

    Args:
        model: 사용할 LLM 모델
            - 기본값: claude-sonnet-4-5-20250929
            - 문자열 형식: "provider:model" (예: "openai:gpt-5")
            - BaseChatModel 인스턴스도 직접 전달 가능

        tools: 에이전트가 사용할 커스텀 도구 목록
            - 기본 도구(계획, 파일, 서브에이전트)는 자동으로 추가됨
            - BaseTool, Callable, dict 형태 모두 지원

        system_prompt: 기본 프롬프트 앞에 추가할 커스텀 시스템 지시사항
            - 문자열: 기본 프롬프트와 연결
            - SystemMessage: content_blocks에 추가

        middleware: 표준 미들웨어 스택 이후에 적용할 추가 미들웨어
            - 기본 스택: TodoList, Filesystem, SubAgent, Summarization,
              AnthropicPromptCaching, PatchToolCalls

        subagents: 사용할 서브에이전트 목록
            - 필수 키: name, description, prompt
            - 선택 키: tools, model, middleware, skills

        skills: 스킬 소스 경로 목록 (예: ["/skills/user/", "/skills/project/"])
            - POSIX 경로 규칙 사용 (슬래시)
            - 같은 이름의 스킬은 마지막 것이 우선

        memory: 메모리 파일(AGENTS.md) 경로 목록
            - 에이전트 시작 시 로드되어 시스템 프롬프트에 추가

        response_format: 에이전트의 구조화된 출력 형식

        context_schema: Deep Agent의 컨텍스트 스키마 타입

        checkpointer: 실행 간 에이전트 상태 유지를 위한 체크포인터

        store: 영구 저장소 (StoreBackend 사용 시 필수)

        backend: 파일 저장 및 실행 백엔드
            - BackendProtocol 인스턴스 또는 팩토리 함수
            - 기본값: StateBackend

        interrupt_on: 도구별 인터럽트 설정 매핑
            - 예: {"edit_file": True}는 모든 편집 전 일시 중지

        debug: 디버그 모드 활성화 여부

        name: 에이전트 이름

        cache: 에이전트 캐시

    Returns:
        CompiledStateGraph: 구성된 Deep Agent (재귀 제한: 1000)

    Raises:
        ValueError: 지원되지 않는 모델 형식이 전달된 경우

    Note:
        - 미들웨어는 순서대로 적용되며 순서가 동작에 영향을 미칩니다.
        - 서브에이전트도 동일한 미들웨어 스택을 기본으로 사용합니다.
        - execute 도구는 SandboxBackendProtocol 미구현 백엔드에서 오류를 반환합니다.
    """
    # ========================================
    # 1. 모델 초기화
    # ========================================
    # model이 None이면 기본 Claude Sonnet 4.5 사용
    # 문자열이면 init_chat_model로 파싱 (예: "openai:gpt-4")
    if model is None:
        model = get_default_model()
    elif isinstance(model, str):
        model = init_chat_model(model)

    # 모델 프로필에 따른 요약(summarization) 기본값 계산
    # - 모델별로 최적화된 컨텍스트 관리 설정 결정
    summarization_defaults = _compute_summarization_defaults(model)

    # 백엔드 초기화: 기본값은 StateBackend 팩토리
    # - StateBackend: 메모리 기반 상태 관리
    # - FilesystemBackend: 디스크 기반 파일 저장
    backend = backend if backend is not None else (lambda rt: StateBackend(rt))

    # ========================================
    # 2. 범용 서브에이전트(General-Purpose) 미들웨어 스택 구성
    # ========================================
    # 범용 서브에이전트는 task 도구로 호출되는 기본 서브에이전트
    gp_middleware: list[AgentMiddleware] = [
        TodoListMiddleware(),  # 할 일 목록 관리 기능
        FilesystemMiddleware(backend=backend),  # 파일 시스템 접근 도구
        SummarizationMiddleware(  # 컨텍스트 요약 및 압축
            model=model,
            backend=backend,
            trigger=summarization_defaults["trigger"],
            keep=summarization_defaults["keep"],
            trim_tokens_to_summarize=None,
            truncate_args_settings=summarization_defaults["truncate_args_settings"],
        ),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),  # 프롬프트 캐싱 최적화
        PatchToolCallsMiddleware(),  # 도구 호출 패치 (호환성)
    ]
    # 스킬이 지정된 경우 SkillsMiddleware 추가
    if skills is not None:
        gp_middleware.append(SkillsMiddleware(backend=backend, sources=skills))
    # 인터럽트 설정이 있으면 Human-in-the-Loop 미들웨어 추가
    if interrupt_on is not None:
        gp_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    # 범용 서브에이전트 스펙 생성
    # - GENERAL_PURPOSE_SUBAGENT: 기본 name, description, prompt 포함
    # - 메인 에이전트와 동일한 모델 및 도구 사용
    general_purpose_spec: SubAgent = {
        **GENERAL_PURPOSE_SUBAGENT,
        "model": model,
        "tools": tools or [],
        "middleware": gp_middleware,
    }

    # ========================================
    # 3. 사용자 정의 서브에이전트 처리
    # ========================================
    # 사용자가 제공한 서브에이전트에 기본값(model, tools, middleware) 적용
    processed_subagents: list[SubAgent | CompiledSubAgent] = []
    for spec in subagents or []:
        if "runnable" in spec:
            # CompiledSubAgent: 이미 컴파일된 에이전트는 그대로 사용
            processed_subagents.append(spec)
        else:
            # SubAgent: 기본값 채우기 및 기본 미들웨어 스택 적용
            # 모델이 지정되지 않으면 메인 에이전트의 모델 사용
            subagent_model = spec.get("model", model)
            if isinstance(subagent_model, str):
                subagent_model = init_chat_model(subagent_model)

            # 서브에이전트 미들웨어 구성:
            # 기본 스택 + 스킬(지정시) + 사용자 미들웨어
            subagent_summarization_defaults = _compute_summarization_defaults(subagent_model)
            subagent_middleware: list[AgentMiddleware] = [
                TodoListMiddleware(),
                FilesystemMiddleware(backend=backend),
                SummarizationMiddleware(
                    model=subagent_model,
                    backend=backend,
                    trigger=subagent_summarization_defaults["trigger"],
                    keep=subagent_summarization_defaults["keep"],
                    trim_tokens_to_summarize=None,
                    truncate_args_settings=subagent_summarization_defaults["truncate_args_settings"],
                ),
                AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
                PatchToolCallsMiddleware(),
            ]
            # 서브에이전트 전용 스킬 추가
            subagent_skills = spec.get("skills")
            if subagent_skills:
                subagent_middleware.append(SkillsMiddleware(backend=backend, sources=subagent_skills))
            # 사용자 정의 미들웨어 추가
            subagent_middleware.extend(spec.get("middleware", []))

            # 처리된 서브에이전트 스펙 생성
            processed_spec: SubAgent = {
                **spec,
                "model": subagent_model,
                "tools": spec.get("tools", tools or []),  # 도구 미지정 시 메인 에이전트 도구 상속
                "middleware": subagent_middleware,
            }
            processed_subagents.append(processed_spec)

    # 범용 서브에이전트 + 사용자 정의 서브에이전트 결합
    all_subagents: list[SubAgent | CompiledSubAgent] = [general_purpose_spec, *processed_subagents]

    # ========================================
    # 4. 메인 에이전트 미들웨어 스택 구성
    # ========================================
    # 미들웨어 적용 순서가 중요함 - 순서대로 체인 형태로 실행됨
    deepagent_middleware: list[AgentMiddleware] = [
        TodoListMiddleware(),  # 1. 할 일 목록 관리 (write_todos 도구 제공)
    ]
    # 메모리 파일(AGENTS.md)이 지정된 경우 시스템 프롬프트에 추가
    if memory is not None:
        deepagent_middleware.append(MemoryMiddleware(backend=backend, sources=memory))
    # 스킬 경로가 지정된 경우 스킬 로딩 미들웨어 추가
    if skills is not None:
        deepagent_middleware.append(SkillsMiddleware(backend=backend, sources=skills))

    # 핵심 미들웨어 확장
    deepagent_middleware.extend(
        [
            FilesystemMiddleware(backend=backend),  # 2. 파일 시스템 도구 (ls, read, write, edit, glob, grep)
            SubAgentMiddleware(  # 3. 서브에이전트 관리 (task 도구 제공)
                backend=backend,
                subagents=all_subagents,
            ),
            SummarizationMiddleware(  # 4. 컨텍스트 요약 (긴 대화 압축)
                model=model,
                backend=backend,
                trigger=summarization_defaults["trigger"],
                keep=summarization_defaults["keep"],
                trim_tokens_to_summarize=None,
                truncate_args_settings=summarization_defaults["truncate_args_settings"],
            ),
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),  # 5. Anthropic 프롬프트 캐싱
            PatchToolCallsMiddleware(),  # 6. 도구 호출 패치 (JSON 스키마 호환성)
        ]
    )
    # 사용자 정의 미들웨어 추가 (기본 스택 이후)
    if middleware:
        deepagent_middleware.extend(middleware)
    # Human-in-the-Loop 미들웨어는 마지막에 추가 (최종 인터럽트 제어)
    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    # ========================================
    # 5. 시스템 프롬프트 결합
    # ========================================
    # 사용자 정의 프롬프트와 BASE_AGENT_PROMPT를 결합
    # - None: 기본 프롬프트만 사용
    # - SystemMessage: content_blocks에 추가
    # - str: 단순 문자열 연결
    if system_prompt is None:
        final_system_prompt: str | SystemMessage = BASE_AGENT_PROMPT
    elif isinstance(system_prompt, SystemMessage):
        # SystemMessage 객체인 경우: content_blocks 배열에 기본 프롬프트 추가
        new_content = [
            *system_prompt.content_blocks,
            {"type": "text", "text": f"\n\n{BASE_AGENT_PROMPT}"},
        ]
        final_system_prompt = SystemMessage(content=new_content)
    else:
        # 문자열인 경우: 단순 연결
        final_system_prompt = system_prompt + "\n\n" + BASE_AGENT_PROMPT

    # ========================================
    # 6. 에이전트 생성 및 반환
    # ========================================
    # LangChain의 create_agent 호출 후 재귀 제한 설정
    # - recursion_limit: 1000 (깊은 서브에이전트 체인 허용)
    return create_agent(
        model,
        system_prompt=final_system_prompt,
        tools=tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 1000})
