"""
모듈명: __init__.py
설명: Deep Agents 패키지의 진입점 및 공개 API 정의

이 모듈은 Deep Agents SDK의 주요 컴포넌트를 외부로 노출합니다.
사용자는 이 모듈을 통해 에이전트 생성 및 미들웨어에 접근할 수 있습니다.

공개 API:
    - create_deep_agent: Deep Agent 인스턴스 생성 함수
    - SubAgent: 서브에이전트 정의 타입
    - CompiledSubAgent: 컴파일된 서브에이전트 타입
    - SubAgentMiddleware: 서브에이전트 관리 미들웨어
    - FilesystemMiddleware: 파일 시스템 접근 미들웨어
    - MemoryMiddleware: 메모리(AGENTS.md) 로딩 미들웨어
    - __version__: 패키지 버전 문자열

사용 예시:
    >>> from deepagents import create_deep_agent, __version__
    >>> print(__version__)
    '0.3.11'
    >>> agent = create_deep_agent()
"""

# 버전 정보 임포트
from deepagents._version import __version__

# 핵심 에이전트 생성 함수
from deepagents.graph import create_deep_agent

# 미들웨어 컴포넌트
# - FilesystemMiddleware: 파일 시스템 도구 제공 (ls, read, write, edit, glob, grep)
# - MemoryMiddleware: AGENTS.md 파일을 시스템 프롬프트에 로드
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware

# 서브에이전트 관련 타입 및 미들웨어
# - SubAgent: 서브에이전트 정의를 위한 TypedDict
# - CompiledSubAgent: 이미 컴파일된 에이전트를 래핑하는 TypedDict
# - SubAgentMiddleware: task 도구를 통한 서브에이전트 호출 관리
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware

# 공개 API 정의
# __all__에 포함된 이름만 "from deepagents import *"로 임포트됨
__all__ = [
    "CompiledSubAgent",      # 컴파일된 서브에이전트 타입
    "FilesystemMiddleware",  # 파일 시스템 미들웨어
    "MemoryMiddleware",      # 메모리 로딩 미들웨어
    "SubAgent",              # 서브에이전트 정의 타입
    "SubAgentMiddleware",    # 서브에이전트 관리 미들웨어
    "__version__",           # 패키지 버전
    "create_deep_agent",     # 에이전트 생성 함수
]
