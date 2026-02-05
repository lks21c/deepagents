"""
모듈명: middleware/__init__.py
설명: 에이전트 미들웨어 패키지의 진입점

이 모듈은 Deep Agents 프레임워크에서 사용되는 다양한 미들웨어 클래스들을
내보냅니다. 미들웨어는 에이전트의 요청/응답 파이프라인에 기능을 추가하는
확장 메커니즘입니다.

주요 미들웨어:
    - FilesystemMiddleware: 파일시스템 도구 제공 (ls, read_file, write_file 등)
    - MemoryMiddleware: AGENTS.md 파일에서 에이전트 메모리/컨텍스트 로드
    - SkillsMiddleware: 온디맨드 스킬/워크플로우 기능 제공
    - SubAgentMiddleware: 서브에이전트 위임 및 조율 지원
    - SummarizationMiddleware: 컨텍스트 윈도우 관리를 위한 대화 요약

사용 예시:
    ```python
    from deepagents.middleware import FilesystemMiddleware, MemoryMiddleware

    # 에이전트 생성 시 미들웨어 적용
    agent = create_agent(
        middleware=[
            FilesystemMiddleware(),
            MemoryMiddleware(backend=backend, sources=["~/.deepagents/AGENTS.md"]),
        ]
    )
    ```
"""

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware
from deepagents.middleware.summarization import SummarizationMiddleware

# 외부로 노출되는 공개 API 목록
__all__ = [
    "CompiledSubAgent",       # 컴파일된 서브에이전트 래퍼
    "FilesystemMiddleware",   # 파일시스템 도구 미들웨어
    "MemoryMiddleware",       # 에이전트 메모리 로딩 미들웨어
    "SkillsMiddleware",       # 스킬/워크플로우 미들웨어
    "SubAgent",               # 서브에이전트 정의 클래스
    "SubAgentMiddleware",     # 서브에이전트 조율 미들웨어
    "SummarizationMiddleware",  # 대화 요약 미들웨어
]
