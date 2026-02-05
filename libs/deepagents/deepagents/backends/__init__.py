"""
모듈명: __init__.py
설명: 플러그형 파일 저장소를 위한 메모리 백엔드 패키지

이 패키지는 다양한 저장소 위치(상태, 파일시스템, 데이터베이스 등)에
파일을 저장하고 통일된 인터페이스를 제공하는 백엔드 구현체들을 포함합니다.

주요 클래스:
    - BackendProtocol: 모든 백엔드가 따라야 하는 기본 프로토콜 인터페이스
    - CompositeBackend: 경로 접두사 기반으로 여러 백엔드에 작업을 라우팅
    - FilesystemBackend: 실제 파일시스템에 직접 읽기/쓰기
    - LocalShellBackend: 로컬 셸 명령 실행 지원 백엔드
    - StateBackend: LangGraph 상태에 파일을 저장하는 백엔드
    - StoreBackend: 영구 저장소에 파일을 저장하는 백엔드

사용 예시:
    >>> from deepagents.backends import FilesystemBackend, CompositeBackend
    >>> fs_backend = FilesystemBackend(root_dir="/workspace")
    >>> content = fs_backend.read("/config.json")
"""

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend

__all__ = [
    "BackendProtocol",
    "CompositeBackend",
    "FilesystemBackend",
    "LocalShellBackend",
    "StateBackend",
    "StoreBackend",
]
