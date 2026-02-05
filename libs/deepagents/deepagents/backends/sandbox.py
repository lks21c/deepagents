"""
모듈명: sandbox.py
설명: execute()만 추상 메서드로 하는 기본 샌드박스 구현

이 모듈은 execute()를 통해 실행되는 셸 명령으로 모든
SandboxBackendProtocol 메서드를 구현하는 기본 클래스를 제공합니다.
구체적인 구현은 execute() 메서드만 구현하면 됩니다.

또한 샌드박스 생명주기 관리(list, create, delete)를 위한 서드파티 SDK
구현용 SandboxProvider 추상 기본 클래스도 정의합니다.

주요 클래스:
    - BaseSandbox: execute()만 구현하면 되는 샌드박스 기본 클래스
    - SandboxProvider: 샌드박스 생명주기 관리를 위한 추상 기본 클래스
    - SandboxInfo: 샌드박스 메타데이터 정보를 담는 TypedDict
    - SandboxListResponse: 페이지네이션된 샌드박스 목록 응답

설계 철학:
    - 셸 명령 기반 구현으로 다양한 실행 환경 지원
    - 부분 성공(partial success) 패턴으로 안정적인 파일 작업
    - 타입 안전성을 위한 제네릭 메타데이터 지원
"""

from __future__ import annotations

import asyncio
import base64
import json
import shlex
from abc import ABC, abstractmethod
from typing import Any, Generic, NotRequired, TypeVar

from typing_extensions import TypedDict

from deepagents.backends.protocol import (
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)

# 프로바이더별 메타데이터를 위한 타입 변수
MetadataT = TypeVar("MetadataT", covariant=True)
"""샌드박스 메타데이터용 타입 변수.

프로바이더는 샌드박스 메타데이터의 구조를 지정하기 위해
자체 TypedDict를 정의할 수 있으며, 이를 통해 메타데이터 필드에
타입 안전하게 접근할 수 있습니다.

사용 예시:
    ```python
    class ProviderMetadata(TypedDict, total=False):
        status: Literal["running", "stopped"]
        created_at: str
        template: str

    class MyProvider(SandboxProvider[ProviderMetadata]):
        def list(
            self, *, cursor=None, **kwargs: Any
        ) -> SandboxListResponse[ProviderMetadata]:
            # 필요에 따라 kwargs 추출
            status = kwargs.get("status")
            ...
    ```
"""


class SandboxInfo(TypedDict, Generic[MetadataT]):
    """단일 샌드박스 인스턴스의 메타데이터.

    이 경량 구조체는 목록 작업에서 반환되며, 전체 연결 없이도
    샌드박스에 대한 기본 정보를 제공합니다.

    타입 파라미터:
        MetadataT: 메타데이터 필드의 타입. 프로바이더는 타입 안전한
            메타데이터 접근을 위해 TypedDict를 정의해야 함.

    속성:
        sandbox_id: 샌드박스 인스턴스의 고유 식별자.
        metadata: 선택적 프로바이더별 메타데이터 (예: 생성 시간, 상태,
            리소스 제한, 템플릿 정보). 구조는 프로바이더가 정의.

    사용 예시:
        ```python
        # 기본 dict[str, Any] 사용
        info: SandboxInfo = {
            "sandbox_id": "sb_abc123",
            "metadata": {"status": "running", "created_at": "2024-01-15T10:30:00Z", "template": "python-3.11"},
        }


        # 타입이 지정된 메타데이터 사용
        class MyMetadata(TypedDict, total=False):
            status: Literal["running", "stopped"]
            created_at: str


        typed_info: SandboxInfo[MyMetadata] = {
            "sandbox_id": "sb_abc123",
            "metadata": {"status": "running", "created_at": "2024-01-15T10:30:00Z"},
        }
        ```
    """

    sandbox_id: str
    metadata: NotRequired[MetadataT]


class SandboxListResponse(TypedDict, Generic[MetadataT]):
    """샌드박스 목록 작업의 페이지네이션된 응답.

    이 구조체는 대규모 샌드박스 컬렉션을 효율적으로 탐색하기 위한
    커서 기반 페이지네이션을 지원합니다.

    타입 파라미터:
        MetadataT: SandboxInfo 항목의 메타데이터 필드 타입.

    속성:
        items: 현재 페이지의 샌드박스 메타데이터 객체 목록.
        cursor: 다음 페이지를 조회하기 위한 불투명 연속 토큰.
            None은 더 이상 페이지가 없음을 나타냄. 클라이언트는
            이것을 불투명 문자열로 취급하고 후속 list() 호출에 전달해야 함.

    사용 예시:
        ```python
        response: SandboxListResponse[MyMetadata] = {
            "items": [{"sandbox_id": "sb_001", "metadata": {"status": "running"}}, {"sandbox_id": "sb_002", "metadata": {"status": "stopped"}}],
            "cursor": "eyJvZmZzZXQiOjEwMH0=",
        }

        # 다음 페이지 가져오기
        next_response = provider.list(cursor=response["cursor"])
        ```
    """

    items: list[SandboxInfo[MetadataT]]
    cursor: str | None


class SandboxProvider(ABC, Generic[MetadataT]):
    """서드파티 샌드박스 프로바이더 구현을 위한 추상 기본 클래스.

    샌드박스 프로바이더의 생명주기 관리 인터페이스를 정의합니다.
    구현은 각 SDK와 통합하여 표준화된 샌드박스 생명주기 작업
    (list, get_or_create, delete)을 제공해야 합니다.

    구현은 호환성을 유지하면서 타입 안전한 API를 제공하기 위해
    기본값이 있는 키워드 전용 인자로 프로바이더별 파라미터를 추가할 수 있습니다.

    동기/비동기 규약:
        LangChain 규약을 따라, 프로바이더는 가능하면 같은 네임스페이스에
        동기와 비동기 메서드를 모두 제공해야 합니다 (성능에 영향 없음).
        (예: 한 클래스에 list()와 alist() 모두 제공)
        기본 비동기 구현은 스레드 풀을 통해 동기 메서드에 위임합니다.
        프로바이더는 필요시 최적화된 비동기 구현을 위해 비동기 메서드를
        오버라이드할 수 있습니다.

        또는 성능 최적화가 필요한 경우, 별도 구현으로 분리할 수 있습니다
        (예: MySyncProvider와 MyAsyncProvider). 이 경우, 구현되지 않은
        메서드는 명확한 안내와 함께 NotImplementedError를 발생시켜야 합니다
        (예: "이 프로바이더는 비동기 작업만 지원합니다. 'await provider.alist()'를
        사용하거나 동기 코드용 MySyncProvider로 전환하세요").

    구현 예시:
        ```python
        class CustomMetadata(TypedDict, total=False):
            status: Literal["running", "stopped"]
            template: str
            created_at: str


        class CustomSandboxProvider(SandboxProvider[CustomMetadata]):
            def list(
                self, *, cursor=None, status: Literal["running", "stopped"] | None = None, template_id: str | None = None, **kwargs: Any
            ) -> SandboxListResponse[CustomMetadata]:
                # IDE 자동완성이 가능한 타입 안전 파라미터
                # ... 프로바이더 API 쿼리
                return {"items": [...], "cursor": None}

            def get_or_create(
                self, *, sandbox_id=None, template_id: str = "default", timeout_minutes: int | None = None, **kwargs: Any
            ) -> SandboxBackendProtocol:
                # IDE 자동완성이 가능한 타입 안전 파라미터
                return CustomSandbox(sandbox_id or self._create_new(), template_id)

            def delete(self, *, sandbox_id: str, force: bool = False, **kwargs: Any) -> None:
                # 구현
                self._client.delete(sandbox_id, force=force)
        ```
    """

    @abstractmethod
    def list(
        self,
        *,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> SandboxListResponse[MetadataT]:
        """사용 가능한 샌드박스를 선택적 필터링과 페이지네이션으로 나열합니다.

        인자:
            cursor: 이전 list() 호출의 선택적 연속 토큰.
                처음부터 시작하려면 None 전달. 커서는 불투명하고
                프로바이더별로 다름; 클라이언트는 이를 파싱하거나 수정하면 안 됨.
            **kwargs: 프로바이더별 필터 파라미터. 구현은 타입 안전성을 위해
                기본값이 있는 명명된 키워드 전용 파라미터로 노출해야 함.
                일반적인 예: 상태 필터, 생성 시간 범위, 템플릿 필터, 소유자 필터.

        반환값:
            다음을 포함하는 SandboxListResponse:
                - items: 현재 페이지의 샌드박스 메타데이터 목록
                - cursor: 다음 페이지 토큰, 마지막 페이지면 None

        사용 예시:
            ```python
            # 첫 번째 페이지
            response = provider.list()
            for sandbox in response["items"]:
                print(sandbox["sandbox_id"])

            # 다음 페이지가 있으면 가져오기
            if response["cursor"]:
                next_response = provider.list(cursor=response["cursor"])

            # 필터 사용 (프로바이더가 지원하는 경우)
            running = provider.list(status="running")
            ```
        """

    @abstractmethod
    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """기존 샌드박스를 가져오거나 새로 생성합니다.

        이 메서드는 sandbox_id가 제공되면 기존 샌드박스에 대한 연결을 조회하고,
        sandbox_id가 None이면 새 샌드박스 인스턴스를 생성합니다.
        반환된 객체는 SandboxBackendProtocol을 구현하며 모든 샌드박스 작업
        (execute, read, write 등)에 사용할 수 있습니다.

        중요: sandbox_id가 제공되었지만 존재하지 않으면, 새 샌드박스를
        생성하지 않고 오류를 발생시켜야 합니다. sandbox_id가 명시적으로
        None일 때만 새 샌드박스를 생성해야 합니다.

        인자:
            sandbox_id: 조회할 기존 샌드박스의 고유 식별자.
                None이면 새 샌드박스 인스턴스 생성. 새 샌드박스의 ID는
                반환된 객체의 .id 속성으로 접근 가능.
                None이 아닌 값이 제공되었지만 샌드박스가 존재하지 않으면
                오류가 발생함.
            **kwargs: 프로바이더별 생성/연결 파라미터. 구현은 타입 안전성을 위해
                기본값이 있는 명명된 키워드 전용 파라미터로 노출해야 함.
                일반적인 예: template_id, 리소스 제한, 환경 변수, 타임아웃 설정.

        반환값:
            명령 실행, 파일 읽기/쓰기 및 기타 샌드박스 작업을 수행할 수 있는
            SandboxBackendProtocol을 구현하는 객체.

        예외:
            다음과 같은 오류에 대한 구현별 예외:
                - 샌드박스를 찾을 수 없음 (sandbox_id가 제공되었지만 존재하지 않는 경우)
                - 권한 부족
                - 리소스 제한 초과
                - 잘못된 템플릿 또는 구성

        사용 예시:
            ```python
            # 새 샌드박스 생성
            sandbox = provider.get_or_create(sandbox_id=None, template_id="python-3.11", timeout_minutes=60)
            print(sandbox.id)  # "sb_new123"

            # 기존 샌드박스에 재연결
            existing = provider.get_or_create(sandbox_id="sb_new123")

            # 샌드박스 사용
            result = sandbox.execute("python --version")
            print(result.output)
            ```
        """

    @abstractmethod
    def delete(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> None:
        """샌드박스 인스턴스를 삭제합니다.

        이 작업은 샌드박스와 모든 관련 데이터를 영구적으로 파괴합니다.
        일반적으로 작업은 되돌릴 수 없습니다.

        멱등성: 이 메서드는 멱등성을 가져야 합니다 - 존재하지 않는 샌드박스에
        delete를 호출해도 오류 없이 성공해야 합니다. 이렇게 하면 정리 코드가
        간단해지고 재시도해도 안전합니다.

        인자:
            sandbox_id: 삭제할 샌드박스의 고유 식별자.
            **kwargs: 프로바이더별 삭제 옵션. 구현은 타입 안전성을 위해
                기본값이 있는 명명된 키워드 전용 파라미터로 노출해야 함.
                일반적인 예: 강제 플래그, 유예 기간, 정리 옵션.

        예외:
            다음과 같은 오류에 대한 구현별 예외:
                - 권한 부족
                - 샌드박스가 잠겨 있거나 사용 중
                - 네트워크 또는 API 오류

        사용 예시:
            ```python
            # 단순 삭제
            provider.delete(sandbox_id="sb_123")

            # 여러 번 호출해도 안전 (멱등성)
            provider.delete(sandbox_id="sb_123")  # 이미 삭제되어도 오류 없음

            # 옵션 사용 (프로바이더가 지원하는 경우)
            provider.delete(sandbox_id="sb_456", force=True)
            ```
        """

    async def alist(
        self,
        *,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> SandboxListResponse[MetadataT]:
        """list()의 비동기 버전.

        기본적으로 동기 list() 메서드를 스레드 풀에서 실행합니다.
        프로바이더는 네이티브 비동기 구현을 위해 이를 오버라이드할 수 있습니다.

        인자:
            cursor: 이전 list() 호출의 선택적 연속 토큰.
            **kwargs: 프로바이더별 필터 파라미터.

        반환값:
            페이지네이션을 위한 items와 cursor를 포함하는 SandboxListResponse.
        """
        return await asyncio.to_thread(self.list, cursor=cursor, **kwargs)

    async def aget_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """get_or_create()의 비동기 버전.

        기본적으로 동기 get_or_create() 메서드를 스레드 풀에서 실행합니다.
        프로바이더는 네이티브 비동기 구현을 위해 이를 오버라이드할 수 있습니다.

        중요: sandbox_id가 제공되었지만 존재하지 않으면, 새 샌드박스를
        생성하지 않고 오류를 발생시켜야 합니다. sandbox_id가 명시적으로
        None일 때만 새 샌드박스를 생성해야 합니다.

        인자:
            sandbox_id: 조회할 기존 샌드박스의 고유 식별자.
                None이면 새 샌드박스 인스턴스 생성. None이 아닌 값이
                제공되었지만 샌드박스가 존재하지 않으면 오류가 발생함.
            **kwargs: 프로바이더별 생성/연결 파라미터.

        반환값:
            SandboxBackendProtocol을 구현하는 객체.
        """
        return await asyncio.to_thread(self.get_or_create, sandbox_id=sandbox_id, **kwargs)

    async def adelete(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> None:
        """delete()의 비동기 버전.

        기본적으로 동기 delete() 메서드를 스레드 풀에서 실행합니다.
        프로바이더는 네이티브 비동기 구현을 위해 이를 오버라이드할 수 있습니다.

        인자:
            sandbox_id: 삭제할 샌드박스의 고유 식별자.
            **kwargs: 프로바이더별 삭제 옵션.
        """
        await asyncio.to_thread(self.delete, sandbox_id=sandbox_id, **kwargs)


# ============================================================================
# 셸 명령 템플릿
# ============================================================================
# 이 템플릿들은 원격 샌드박스에서 셸 명령을 통해 파일 작업을 수행합니다.
# Python 스크립트를 셸에서 실행하여 파일시스템 작업을 수행하고
# 결과를 JSON 형식으로 반환합니다.

# Glob 명령 템플릿: 파일 패턴 매칭을 수행
# base64로 인코딩된 경로와 패턴을 받아 매칭되는 파일 정보를 JSON으로 출력
_GLOB_COMMAND_TEMPLATE = """python3 -c "
import glob
import os
import json
import base64

# base64로 인코딩된 파라미터 디코딩
path = base64.b64decode('{path_b64}').decode('utf-8')
pattern = base64.b64decode('{pattern_b64}').decode('utf-8')

os.chdir(path)
matches = sorted(glob.glob(pattern, recursive=True))
for m in matches:
    stat = os.stat(m)
    result = {{
        'path': m,
        'size': stat.st_size,
        'mtime': stat.st_mtime,
        'is_dir': os.path.isdir(m)
    }}
    print(json.dumps(result))
" 2>/dev/null"""

# 쓰기 명령 템플릿: 대용량 파일의 ARG_MAX 제한을 피하기 위해 heredoc 사용
# ARG_MAX는 명령행 인수의 총 크기를 제한합니다.
# 이전에는 base64로 인코딩된 콘텐츠를 명령 문자열에 직접 삽입했는데,
# base64 확장 후 ~100KB 이상의 파일에서 실패했습니다.
# Heredoc은 인수가 아닌 stdin을 통해 데이터를 전달하여 이를 우회합니다.
# Stdin 형식: JSON 페이로드에 파일 경로와 콘텐츠가 base64로 인코딩됨
_WRITE_COMMAND_TEMPLATE = """python3 -c "
import os
import sys
import base64
import json

# stdin에서 file_path와 content(둘 다 base64 인코딩)를 포함하는 JSON 페이로드 읽기
payload_b64 = sys.stdin.read().strip()
if not payload_b64:
    print('Error: No payload received for write operation', file=sys.stderr)
    sys.exit(1)

try:
    payload = base64.b64decode(payload_b64).decode('utf-8')
    data = json.loads(payload)
    file_path = data['path']
    content = base64.b64decode(data['content']).decode('utf-8')
except Exception as e:
    print(f'Error: Failed to decode write payload: {{e}}', file=sys.stderr)
    sys.exit(1)

# 파일이 이미 존재하는지 확인 (쓰기와 원자적으로)
if os.path.exists(file_path):
    print(f'Error: File \\'{{file_path}}\\' already exists', file=sys.stderr)
    sys.exit(1)

# 필요시 부모 디렉토리 생성
parent_dir = os.path.dirname(file_path) or '.'
os.makedirs(parent_dir, exist_ok=True)

with open(file_path, 'w') as f:
    f.write(content)
" <<'__DEEPAGENTS_EOF__'
{payload_b64}
__DEEPAGENTS_EOF__"""

# 편집 명령 템플릿: ARG_MAX 제한을 피하기 위해 heredoc으로 편집 파라미터 전달
# Stdin 형식: {"path": str, "old": str, "new": str}를 포함하는 base64 인코딩 JSON
# JSON은 모든 파라미터를 번들링하고, base64는 이스케이프 문제 없이
# 임의 콘텐츠(특수 문자, 줄바꿈 등)의 안전한 전송을 보장
_EDIT_COMMAND_TEMPLATE = """python3 -c "
import sys
import base64
import json
import os

# stdin에서 JSON 페이로드 읽기 및 디코딩
payload_b64 = sys.stdin.read().strip()
if not payload_b64:
    print('Error: No payload received for edit operation', file=sys.stderr)
    sys.exit(4)

try:
    payload = base64.b64decode(payload_b64).decode('utf-8')
    data = json.loads(payload)
    file_path = data['path']
    old = data['old']
    new = data['new']
except Exception as e:
    print(f'Error: Failed to decode edit payload: {{e}}', file=sys.stderr)
    sys.exit(4)

# 파일 존재 확인
if not os.path.isfile(file_path):
    sys.exit(3)  # 파일 없음

# 파일 내용 읽기
with open(file_path, 'r') as f:
    text = f.read()

# 발생 횟수 계산
count = text.count(old)

# 문제가 발견되면 오류 코드와 함께 종료
if count == 0:
    sys.exit(1)  # 문자열 없음
elif count > 1 and not {replace_all}:
    sys.exit(2)  # replace_all 없이 다중 발생

# 교체 수행
if {replace_all}:
    result = text.replace(old, new)
else:
    result = text.replace(old, new, 1)

# 파일에 다시 쓰기
with open(file_path, 'w') as f:
    f.write(result)

print(count)
" <<'__DEEPAGENTS_EOF__'
{payload_b64}
__DEEPAGENTS_EOF__"""

# 읽기 명령 템플릿: 오프셋과 제한으로 파일 내용 읽기
# 줄 번호와 함께 포맷팅된 출력 반환
_READ_COMMAND_TEMPLATE = """python3 -c "
import os
import sys

file_path = '{file_path}'
offset = {offset}
limit = {limit}

# 파일 존재 확인
if not os.path.isfile(file_path):
    print('Error: File not found')
    sys.exit(1)

# 파일이 비어있는지 확인
if os.path.getsize(file_path) == 0:
    print('System reminder: File exists but has empty contents')
    sys.exit(0)

# 오프셋과 제한으로 파일 읽기
with open(file_path, 'r') as f:
    lines = f.readlines()

# 오프셋과 제한 적용
start_idx = offset
end_idx = offset + limit
selected_lines = lines[start_idx:end_idx]

# 줄 번호와 함께 포맷팅 (1부터 시작, offset + 1부터 시작)
for i, line in enumerate(selected_lines):
    line_num = offset + i + 1
    # 포맷팅을 위해 후행 줄바꿈 제거 후 다시 추가
    line_content = line.rstrip('\\n')
    print(f'{{line_num:6d}}\\t{{line_content}}')
" 2>&1"""


class BaseSandbox(SandboxBackendProtocol, ABC):
    """execute()를 추상 메서드로 하는 기본 샌드박스 구현.

    이 클래스는 셸 명령을 사용하여 모든 프로토콜 메서드의
    기본 구현을 제공합니다. 하위 클래스는 execute()만 구현하면 됩니다.

    설계 철학:
        - 셸 명령 기반으로 다양한 실행 환경 지원 (Docker, VM, 원격 서버 등)
        - 파일 작업을 Python 스크립트로 구현하여 일관된 동작 보장
        - 부분 성공 패턴으로 안정적인 배치 파일 작업 지원

    하위 클래스가 구현해야 할 추상 메서드:
        - execute(): 샌드박스에서 명령 실행
        - id: 샌드박스의 고유 식별자 속성
        - upload_files(): 파일 업로드
        - download_files(): 파일 다운로드
    """

    @abstractmethod
    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """샌드박스에서 명령을 실행하고 ExecuteResponse를 반환합니다.

        인자:
            command: 실행할 전체 셸 명령 문자열.

        반환값:
            결합된 출력, 종료 코드, 선택적 시그널, 잘림 플래그를 포함하는 ExecuteResponse.
        """
        ...

    def ls_info(self, path: str) -> list[FileInfo]:
        """os.scandir를 사용하여 파일 메타데이터가 포함된 구조화된 목록 반환."""
        cmd = f"""python3 -c "
import os
import json

path = '{path}'

try:
    with os.scandir(path) as it:
        for entry in it:
            result = {{
                'path': os.path.join(path, entry.name),
                'is_dir': entry.is_dir(follow_symlinks=False)
            }}
            print(json.dumps(result))
except FileNotFoundError:
    pass
except PermissionError:
    pass
" 2>/dev/null"""

        result = self.execute(cmd)

        file_infos: list[FileInfo] = []
        for line in result.output.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                file_infos.append({"path": data["path"], "is_dir": data["is_dir"]})
            except json.JSONDecodeError:
                continue

        return file_infos

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers using a single shell command."""
        # Use template for reading file with offset and limit
        cmd = _READ_COMMAND_TEMPLATE.format(file_path=file_path, offset=offset, limit=limit)
        result = self.execute(cmd)

        output = result.output.rstrip()
        exit_code = result.exit_code

        if exit_code != 0 or "Error: File not found" in output:
            return f"Error: File '{file_path}' not found"

        return output

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file. Returns WriteResult; error populated on failure."""
        # Create JSON payload with file path and base64-encoded content
        # This avoids shell injection via file_path and ARG_MAX limits on content
        content_b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        payload = json.dumps({"path": file_path, "content": content_b64})
        payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")

        # Single atomic check + write command
        cmd = _WRITE_COMMAND_TEMPLATE.format(payload_b64=payload_b64)
        result = self.execute(cmd)

        # Check for errors (exit code or error message in output)
        if result.exit_code != 0 or "Error:" in result.output:
            error_msg = result.output.strip() or f"Failed to write file '{file_path}'"
            return WriteResult(error=error_msg)

        # External storage - no files_update needed
        return WriteResult(path=file_path, files_update=None)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences. Returns EditResult."""
        # Create JSON payload with file path, old string, and new string
        # This avoids shell injection via file_path and ARG_MAX limits on strings
        payload = json.dumps({"path": file_path, "old": old_string, "new": new_string})
        payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")

        # Use template for string replacement
        cmd = _EDIT_COMMAND_TEMPLATE.format(payload_b64=payload_b64, replace_all=replace_all)
        result = self.execute(cmd)

        exit_code = result.exit_code
        output = result.output.strip()

        # Map exit codes to error messages
        error_messages = {
            1: f"Error: String not found in file: '{old_string}'",
            2: f"Error: String '{old_string}' appears multiple times. Use replace_all=True to replace all occurrences.",
            3: f"Error: File '{file_path}' not found",
            4: f"Error: Failed to decode edit payload: {output}",
        }
        if exit_code in error_messages:
            return EditResult(error=error_messages[exit_code])
        if exit_code != 0:
            return EditResult(error=f"Error editing file (exit code {exit_code}): {output or 'Unknown error'}")

        count = int(output)
        # External storage - no files_update needed
        return EditResult(path=file_path, files_update=None, occurrences=count)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Structured search results or error string for invalid input."""
        search_path = shlex.quote(path or ".")

        # Build grep command to get structured output
        grep_opts = "-rHnF"  # recursive, with filename, with line number, fixed-strings (literal)

        # Add glob pattern if specified
        glob_pattern = ""
        if glob:
            glob_pattern = f"--include='{glob}'"

        # Escape pattern for shell
        pattern_escaped = shlex.quote(pattern)

        cmd = f"grep {grep_opts} {glob_pattern} -e {pattern_escaped} {search_path} 2>/dev/null || true"
        result = self.execute(cmd)

        output = result.output.rstrip()
        if not output:
            return []

        # Parse grep output into GrepMatch objects
        matches: list[GrepMatch] = []
        for line in output.split("\n"):
            # Format is: path:line_number:text
            parts = line.split(":", 2)
            if len(parts) >= 3:
                matches.append(
                    {
                        "path": parts[0],
                        "line": int(parts[1]),
                        "text": parts[2],
                    }
                )

        return matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Structured glob matching returning FileInfo dicts."""
        # Encode pattern and path as base64 to avoid escaping issues
        pattern_b64 = base64.b64encode(pattern.encode("utf-8")).decode("ascii")
        path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")

        cmd = _GLOB_COMMAND_TEMPLATE.format(path_b64=path_b64, pattern_b64=pattern_b64)
        result = self.execute(cmd)

        output = result.output.strip()
        if not output:
            return []

        # Parse JSON output into FileInfo dicts
        file_infos: list[FileInfo] = []
        for line in output.split("\n"):
            try:
                data = json.loads(line)
                file_infos.append(
                    {
                        "path": data["path"],
                        "is_dir": data["is_dir"],
                    }
                )
            except json.JSONDecodeError:
                continue

        return file_infos

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""

    @abstractmethod
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the sandbox.

        Implementations must support partial success - catch exceptions per-file
        and return errors in FileUploadResponse objects rather than raising.
        """

    @abstractmethod
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the sandbox.

        Implementations must support partial success - catch exceptions per-file
        and return errors in FileDownloadResponse objects rather than raising.
        """
