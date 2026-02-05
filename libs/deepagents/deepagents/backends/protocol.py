"""
모듈명: protocol.py
설명: 플러그형 메모리 백엔드를 위한 프로토콜 정의

이 모듈은 모든 백엔드 구현체가 따라야 하는 BackendProtocol을 정의합니다.
백엔드는 파일을 다양한 위치(상태, 파일시스템, 데이터베이스 등)에 저장하고
파일 작업을 위한 통일된 인터페이스를 제공합니다.

주요 클래스:
    - FileDownloadResponse: 파일 다운로드 작업의 결과를 담는 데이터클래스
    - FileUploadResponse: 파일 업로드 작업의 결과를 담는 데이터클래스
    - FileInfo: 파일 메타데이터를 담는 TypedDict
    - GrepMatch: grep 검색 결과 항목을 담는 TypedDict
    - WriteResult: 파일 쓰기 작업의 결과를 담는 데이터클래스
    - EditResult: 파일 편집 작업의 결과를 담는 데이터클래스
    - ExecuteResponse: 명령 실행 결과를 담는 데이터클래스
    - BackendProtocol: 기본 백엔드 프로토콜 추상 클래스
    - SandboxBackendProtocol: 샌드박스 실행을 지원하는 프로토콜

타입 별칭:
    - FileOperationError: 파일 작업 오류 코드 리터럴 타입
    - BackendFactory: 백엔드 팩토리 함수 타입
    - BACKEND_TYPES: 백엔드 또는 백엔드 팩토리 유니온 타입
"""

import abc
import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypeAlias

from langchain.tools import ToolRuntime
from typing_extensions import TypedDict

FileOperationError = Literal[
    "file_not_found",  # 다운로드: 파일이 존재하지 않음
    "permission_denied",  # 업로드/다운로드: 접근 거부됨
    "is_directory",  # 다운로드: 디렉토리를 파일로 다운로드 시도
    "invalid_path",  # 업로드/다운로드: 경로 문법 오류 (상위 디렉토리 없음, 잘못된 문자)
]
"""
파일 업로드/다운로드 작업을 위한 표준화된 오류 코드입니다.

LLM이 이해하고 잠재적으로 수정할 수 있는 일반적이고 복구 가능한 오류를 나타냅니다:
    - file_not_found: 요청한 파일이 존재하지 않음 (다운로드 시)
    - parent_not_found: 상위 디렉토리가 존재하지 않음 (업로드 시)
    - permission_denied: 작업에 대한 접근이 거부됨
    - is_directory: 디렉토리를 파일로 다운로드 시도함
    - invalid_path: 경로 문법이 잘못되었거나 유효하지 않은 문자 포함
"""


@dataclass
class FileDownloadResponse:
    """
    단일 파일 다운로드 작업의 결과를 담는 데이터클래스입니다.

    배치 작업에서 부분 성공을 허용하도록 설계되었습니다.
    LLM이 파일 작업을 수행하는 사용 사례를 위해 특정 복구 가능한
    조건에 대해 FileOperationError 리터럴을 사용하여 오류를 표준화합니다.

    Attributes:
        path: 요청된 파일 경로. 배치 결과 처리 시 상관관계 파악을 위해 포함되며,
            특히 오류 메시지에 유용합니다.
        content: 성공 시 바이트 형태의 파일 내용, 실패 시 None.
        error: 실패 시 표준화된 오류 코드, 성공 시 None.
            구조화된 LLM 대응 가능한 오류 보고를 위해 FileOperationError 리터럴 사용.

    Examples:
        >>> # 성공 케이스
        >>> FileDownloadResponse(path="/app/config.json", content=b"{...}", error=None)
        >>> # 실패 케이스
        >>> FileDownloadResponse(path="/wrong/path.txt", content=None, error="file_not_found")
    """

    path: str  # 요청된 파일의 절대 경로
    content: bytes | None = None  # 파일 내용 (바이트), 실패 시 None
    error: FileOperationError | None = None  # 오류 코드, 성공 시 None


@dataclass
class FileUploadResponse:
    """
    단일 파일 업로드 작업의 결과를 담는 데이터클래스입니다.

    배치 작업에서 부분 성공을 허용하도록 설계되었습니다.
    LLM이 파일 작업을 수행하는 사용 사례를 위해 특정 복구 가능한
    조건에 대해 FileOperationError 리터럴을 사용하여 오류를 표준화합니다.

    Attributes:
        path: 요청된 파일 경로. 배치 결과 처리 및 명확한 오류 메시지를
            위해 포함됩니다.
        error: 실패 시 표준화된 오류 코드, 성공 시 None.
            구조화된 LLM 대응 가능한 오류 보고를 위해 FileOperationError 리터럴 사용.

    Examples:
        >>> # 성공 케이스
        >>> FileUploadResponse(path="/app/data.txt", error=None)
        >>> # 실패 케이스
        >>> FileUploadResponse(path="/readonly/file.txt", error="permission_denied")
    """

    path: str  # 업로드 대상 파일의 절대 경로
    error: FileOperationError | None = None  # 오류 코드, 성공 시 None


class FileInfo(TypedDict):
    """
    구조화된 파일 목록 정보를 담는 TypedDict입니다.

    모든 백엔드에서 사용되는 최소 계약입니다. "path" 필드만 필수이며,
    다른 필드는 백엔드에 따라 최선의 노력으로 제공되고 없을 수 있습니다.

    Attributes:
        path: 파일의 절대 경로 (필수)
        is_dir: 디렉토리인 경우 True (선택)
        size: 파일 크기(바이트), 근사값일 수 있음 (선택)
        modified_at: ISO 형식 타임스탬프, 알려진 경우 (선택)
    """

    path: str  # 파일의 절대 경로 (필수 필드)
    is_dir: NotRequired[bool]  # 디렉토리 여부 (선택)
    size: NotRequired[int]  # 파일 크기(바이트), 근사값 (선택)
    modified_at: NotRequired[str]  # ISO 타임스탬프 (선택)


class GrepMatch(TypedDict):
    """
    구조화된 grep 검색 결과 항목을 담는 TypedDict입니다.

    파일 내 텍스트 패턴 검색 결과를 표현합니다.

    Attributes:
        path: 매칭이 발견된 파일의 절대 경로
        line: 매칭이 발견된 라인 번호 (1부터 시작)
        text: 매칭이 포함된 전체 라인 텍스트
    """

    path: str  # 매칭 파일의 절대 경로
    line: int  # 라인 번호 (1-indexed)
    text: str  # 매칭된 라인의 전체 텍스트


@dataclass
class WriteResult:
    """
    백엔드 쓰기 작업의 결과를 담는 데이터클래스입니다.

    Attributes:
        error: 실패 시 오류 메시지, 성공 시 None.
        path: 작성된 파일의 절대 경로, 실패 시 None.
        files_update: 체크포인트 백엔드용 상태 업데이트 딕셔너리, 외부 저장소는 None.
            체크포인트 백엔드는 LangGraph 상태를 위해 {file_path: file_data}로 채웁니다.
            외부 백엔드는 None으로 설정합니다 (디스크/S3/데이터베이스 등에 이미 저장됨).

    Examples:
        >>> # 체크포인트 저장소
        >>> WriteResult(path="/f.txt", files_update={"/f.txt": {...}})
        >>> # 외부 저장소
        >>> WriteResult(path="/f.txt", files_update=None)
        >>> # 오류 발생
        >>> WriteResult(error="File exists")
    """

    error: str | None = None  # 오류 메시지, 성공 시 None
    path: str | None = None  # 작성된 파일의 절대 경로
    files_update: dict[str, Any] | None = None  # 상태 업데이트 딕셔너리


@dataclass
class EditResult:
    """
    백엔드 편집 작업의 결과를 담는 데이터클래스입니다.

    Attributes:
        error: 실패 시 오류 메시지, 성공 시 None.
        path: 편집된 파일의 절대 경로, 실패 시 None.
        files_update: 체크포인트 백엔드용 상태 업데이트 딕셔너리, 외부 저장소는 None.
            체크포인트 백엔드는 LangGraph 상태를 위해 {file_path: file_data}로 채웁니다.
            외부 백엔드는 None으로 설정합니다 (디스크/S3/데이터베이스 등에 이미 저장됨).
        occurrences: 수행된 교체 횟수, 실패 시 None.

    Examples:
        >>> # 체크포인트 저장소
        >>> EditResult(path="/f.txt", files_update={"/f.txt": {...}}, occurrences=1)
        >>> # 외부 저장소
        >>> EditResult(path="/f.txt", files_update=None, occurrences=2)
        >>> # 오류 발생
        >>> EditResult(error="File not found")
    """

    error: str | None = None  # 오류 메시지, 성공 시 None
    path: str | None = None  # 편집된 파일의 절대 경로
    files_update: dict[str, Any] | None = None  # 상태 업데이트 딕셔너리
    occurrences: int | None = None  # 교체 횟수


class BackendProtocol(abc.ABC):
    """
    플러그형 메모리 백엔드를 위한 프로토콜 추상 기본 클래스입니다.

    백엔드는 파일을 다양한 위치(상태, 파일시스템, 데이터베이스 등)에 저장하고
    파일 작업을 위한 통일된 인터페이스를 제공합니다.

    모든 파일 데이터는 다음 구조의 딕셔너리로 표현됩니다:
        {
            "content": list[str],  # 텍스트 콘텐츠 라인들
            "created_at": str,     # ISO 형식 타임스탬프
            "modified_at": str,    # ISO 형식 타임스탬프
        }

    주요 메서드:
        - ls_info: 디렉토리 내 파일 목록 조회
        - read: 파일 내용 읽기
        - write: 새 파일 생성
        - edit: 기존 파일 편집
        - grep_raw: 파일 내 텍스트 검색
        - glob_info: 글롭 패턴으로 파일 찾기
        - upload_files: 여러 파일 업로드
        - download_files: 여러 파일 다운로드
    """

    def ls_info(self, path: str) -> list["FileInfo"]:
        """
        디렉토리 내 모든 파일 목록을 메타데이터와 함께 반환합니다.

        Args:
            path: 목록을 조회할 디렉토리의 절대 경로. '/'로 시작해야 함.

        Returns:
            파일 메타데이터를 담은 FileInfo 딕셔너리 목록:
                - path (필수): 파일의 절대 경로
                - is_dir (선택): 디렉토리인 경우 True
                - size (선택): 파일 크기(바이트)
                - modified_at (선택): ISO 8601 형식 타임스탬프
        """

    async def als_info(self, path: str) -> list["FileInfo"]:
        """ls_info의 비동기 버전입니다."""
        return await asyncio.to_thread(self.ls_info, path)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """
        파일 내용을 라인 번호와 함께 읽습니다.

        Args:
            file_path: 읽을 파일의 절대 경로. '/'로 시작해야 함.
            offset: 읽기 시작할 라인 번호 (0-indexed). 기본값: 0.
            limit: 읽을 최대 라인 수. 기본값: 2000.

        Returns:
            라인 번호가 포함된 형식의 파일 내용 문자열 (cat -n 형식),
            라인 번호는 1부터 시작. 2000자를 초과하는 라인은 잘립니다.
            파일이 존재하지 않거나 읽을 수 없는 경우 오류 문자열 반환.

        Note:
            - 대용량 파일의 경우 컨텍스트 오버플로우를 피하기 위해 페이지네이션(offset/limit) 사용
            - 첫 스캔: `read(path, limit=100)`으로 파일 구조 확인
            - 추가 읽기: `read(path, offset=100, limit=200)`으로 다음 섹션 읽기
            - 파일 편집 전에 반드시 먼저 읽기
            - 파일이 존재하지만 비어있는 경우 시스템 알림 경고 수신
        """

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """read의 비동기 버전입니다."""
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list["GrepMatch"] | str:
        """
        파일에서 리터럴 텍스트 패턴을 검색합니다.

        Args:
            pattern: 검색할 리터럴 문자열 (정규식이 아님).
                파일 내용에서 정확한 부분 문자열 매칭을 수행합니다.
                예: "TODO"는 "TODO"를 포함하는 모든 라인과 매칭

            path: 검색할 디렉토리 경로 (선택).
                None인 경우 현재 작업 디렉토리에서 검색.
                예: "/workspace/src"

            glob: 검색할 파일을 필터링하는 글롭 패턴 (선택).
                내용이 아닌 파일명/경로로 필터링.
                표준 글롭 와일드카드 지원:
                    - `*` 파일명 내 모든 문자와 매칭
                    - `**` 모든 디렉토리를 재귀적으로 매칭
                    - `?` 단일 문자 매칭
                    - `[abc]` 집합 내 한 문자 매칭

        Examples:
            - "*.py" - Python 파일만 검색
            - "**/*.txt" - 모든 .txt 파일을 재귀적으로 검색
            - "src/**/*.js" - src/ 아래의 JS 파일 검색
            - "test[0-9].txt" - test0.txt, test1.txt 등 검색

        Returns:
            성공 시: 구조화된 결과가 담긴 list[GrepMatch]:
                - path: 파일의 절대 경로
                - line: 라인 번호 (1부터 시작)
                - text: 매칭을 포함하는 전체 라인 내용

            오류 시: 오류 메시지 문자열 (예: 유효하지 않은 경로, 권한 거부)
        """

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list["GrepMatch"] | str:
        """grep_raw의 비동기 버전입니다."""
        return await asyncio.to_thread(self.grep_raw, pattern, path, glob)

    def glob_info(self, pattern: str, path: str = "/") -> list["FileInfo"]:
        """
        글롭 패턴과 매칭되는 파일을 찾습니다.

        Args:
            pattern: 파일 경로와 매칭할 와일드카드가 포함된 글롭 패턴.
                표준 글롭 문법 지원:
                    - `*` 파일명/디렉토리 내 모든 문자와 매칭
                    - `**` 모든 디렉토리를 재귀적으로 매칭
                    - `?` 단일 문자 매칭
                    - `[abc]` 집합 내 한 문자 매칭

            path: 검색 시작 기본 디렉토리. 기본값: "/" (루트).
                패턴은 이 경로를 기준으로 적용됩니다.

        Returns:
            매칭된 파일 정보를 담은 FileInfo 목록
        """

    async def aglob_info(self, pattern: str, path: str = "/") -> list["FileInfo"]:
        """glob_info의 비동기 버전입니다."""
        return await asyncio.to_thread(self.glob_info, pattern, path)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """
        파일시스템에 새 파일을 생성하고 내용을 씁니다. 파일이 이미 존재하면 오류.

        Args:
            file_path: 파일을 생성할 절대 경로. '/'로 시작해야 함.
            content: 파일에 쓸 문자열 내용.

        Returns:
            WriteResult: 쓰기 작업 결과 (경로, 오류, 상태 업데이트 정보 포함)
        """

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """write의 비동기 버전입니다."""
        return await asyncio.to_thread(self.write, file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """
        기존 파일에서 정확한 문자열 교체를 수행합니다.

        Args:
            file_path: 편집할 파일의 절대 경로. '/'로 시작해야 함.
            old_string: 검색하여 교체할 정확한 문자열.
                공백과 들여쓰기를 포함하여 정확히 일치해야 함.
            new_string: old_string을 대체할 문자열.
                old_string과 달라야 함.
            replace_all: True인 경우 모든 발생을 교체. False(기본값)인 경우
                old_string이 파일 내에서 고유해야 하며, 그렇지 않으면 편집 실패.

        Returns:
            EditResult: 편집 작업 결과 (경로, 오류, 교체 횟수 정보 포함)
        """

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """edit의 비동기 버전입니다."""
        return await asyncio.to_thread(self.edit, file_path, old_string, new_string, replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """
        여러 파일을 샌드박스에 업로드합니다.

        이 API는 개발자가 직접 사용하거나 커스텀 도구를 통해
        LLM에 노출하여 사용할 수 있도록 설계되었습니다.

        Args:
            files: 업로드할 (경로, 내용) 튜플 목록.

        Returns:
            입력 파일당 하나씩 FileUploadResponse 객체 목록.
            응답 순서는 입력 순서와 일치합니다 (response[i]는 files[i]에 대응).
            파일별 성공/실패를 확인하려면 error 필드를 검사하세요.

        Examples:
            ```python
            responses = sandbox.upload_files(
                [
                    ("/app/config.json", b"{...}"),
                    ("/app/data.txt", b"content"),
                ]
            )
            ```
        """

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """upload_files의 비동기 버전입니다."""
        return await asyncio.to_thread(self.upload_files, files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """
        샌드박스에서 여러 파일을 다운로드합니다.

        이 API는 개발자가 직접 사용하거나 커스텀 도구를 통해
        LLM에 노출하여 사용할 수 있도록 설계되었습니다.

        Args:
            paths: 다운로드할 파일 경로 목록.

        Returns:
            입력 경로당 하나씩 FileDownloadResponse 객체 목록.
            응답 순서는 입력 순서와 일치합니다 (response[i]는 paths[i]에 대응).
            파일별 성공/실패를 확인하려면 error 필드를 검사하세요.
        """

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """download_files의 비동기 버전입니다."""
        return await asyncio.to_thread(self.download_files, paths)


@dataclass
class ExecuteResponse:
    """
    코드 실행 결과를 담는 데이터클래스입니다.

    LLM 소비에 최적화된 단순화된 스키마입니다.

    Attributes:
        output: 실행된 명령의 stdout과 stderr을 결합한 출력.
        exit_code: 프로세스 종료 코드. 0은 성공, 0이 아닌 값은 실패를 나타냄.
        truncated: 백엔드 제한으로 인해 출력이 잘렸는지 여부.
    """

    output: str
    """실행된 명령의 stdout과 stderr을 결합한 출력입니다."""

    exit_code: int | None = None
    """프로세스 종료 코드입니다. 0은 성공, 0이 아닌 값은 실패를 나타냅니다."""

    truncated: bool = False
    """백엔드 제한으로 인해 출력이 잘렸는지 여부입니다."""


class SandboxBackendProtocol(BackendProtocol):
    """
    격리된 런타임을 가진 샌드박스 백엔드를 위한 프로토콜입니다.

    샌드박스 백엔드는 격리된 환경(예: 별도 프로세스, 컨테이너)에서 실행되며
    정의된 인터페이스를 통해 통신합니다.

    BackendProtocol을 상속하며 추가로 명령 실행 기능을 제공합니다.

    추가 메서드:
        - execute: 샌드박스 내에서 셸 명령 실행
        - aexecute: execute의 비동기 버전

    추가 속성:
        - id: 샌드박스 백엔드 인스턴스의 고유 식별자
    """

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """
        프로세스에서 명령을 실행합니다.

        LLM 소비에 최적화된 단순화된 인터페이스입니다.

        Args:
            command: 실행할 전체 셸 명령 문자열.

        Returns:
            ExecuteResponse: 결합된 출력, 종료 코드, 선택적 시그널, 잘림 플래그를 포함.
        """

    async def aexecute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """execute의 비동기 버전입니다."""
        return await asyncio.to_thread(self.execute, command)

    @property
    def id(self) -> str:
        """샌드박스 백엔드 인스턴스의 고유 식별자입니다."""


# 타입 별칭: ToolRuntime을 받아 BackendProtocol을 반환하는 팩토리 함수
BackendFactory: TypeAlias = Callable[[ToolRuntime], BackendProtocol]

# 백엔드 타입: 직접 백엔드 인스턴스 또는 백엔드 팩토리 함수
BACKEND_TYPES = BackendProtocol | BackendFactory
