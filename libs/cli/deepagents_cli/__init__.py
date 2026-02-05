"""
모듈명: __init__.py
설명: Deep Agents CLI - 대화형 AI 코딩 어시스턴트

이 패키지는 Deep Agents 프레임워크의 CLI(명령줄 인터페이스) 도구를 제공합니다.
Textual 기반의 TUI(터미널 사용자 인터페이스)를 통해 AI 에이전트와
대화형으로 소통하며 코딩 작업을 수행할 수 있습니다.

주요 기능:
- 대화형 AI 코딩 어시스턴트
- 파일 읽기/쓰기/편집 도구
- 셸 명령 실행
- 웹 검색 및 URL 가져오기
- 메모리 및 스킬 시스템

내보내기:
- __version__: 패키지 버전
- cli_main: CLI 메인 진입점
"""

from deepagents_cli._version import __version__
from deepagents_cli.main import cli_main

__all__ = [
    "__version__",
    "cli_main",
]
