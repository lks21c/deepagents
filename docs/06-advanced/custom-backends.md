# 커스텀 백엔드 구현

> BackendProtocol을 구현하여 사용자 정의 백엔드를 작성하는 방법입니다.

## 개요

Deep Agents의 백엔드 시스템은 파일 저장과 명령 실행을 추상화합니다. 커스텀 백엔드를 구현하면 다양한 저장소(클라우드, 데이터베이스, 컨테이너 등)와 통합할 수 있습니다.

---

## BackendProtocol 인터페이스

**소스 위치**: `libs/deepagents/deepagents/backends/protocol.py`

```python
import abc
from typing import Any

class BackendProtocol(abc.ABC):
    """플러그형 백엔드를 위한 프로토콜

    모든 백엔드 구현체가 구현해야 하는 추상 기본 클래스입니다.
    파일 시스템 작업(읽기, 쓰기, 편집, 검색)을 위한
    표준 인터페이스를 정의합니다.
    """

    @abc.abstractmethod
    def ls_info(self, path: str) -> list["FileInfo"]:
        """디렉토리 내 파일 목록 조회"""

    @abc.abstractmethod
    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """파일 내용 읽기"""

    @abc.abstractmethod
    def write(
        self,
        file_path: str,
        content: str,
    ) -> "WriteResult":
        """새 파일 생성"""

    @abc.abstractmethod
    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> "EditResult":
        """기존 파일 편집"""

    @abc.abstractmethod
    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list["GrepMatch"] | str:
        """텍스트 패턴 검색"""

    @abc.abstractmethod
    def glob_info(
        self,
        pattern: str,
        path: str = "/",
    ) -> list["FileInfo"]:
        """글롭 패턴으로 파일 검색"""

    @abc.abstractmethod
    def upload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list["FileUploadResponse"]:
        """여러 파일 업로드"""

    @abc.abstractmethod
    def download_files(
        self,
        paths: list[str],
    ) -> list["FileDownloadResponse"]:
        """여러 파일 다운로드"""
```

---

## 데이터 타입

### FileInfo

```python
from typing import NotRequired
from typing_extensions import TypedDict

class FileInfo(TypedDict):
    """파일/디렉토리 메타데이터"""

    path: str
    """파일의 절대 경로"""

    is_dir: NotRequired[bool]
    """디렉토리 여부 (기본: False)"""

    size: NotRequired[int]
    """파일 크기 (바이트)"""

    modified_at: NotRequired[str]
    """수정 시각 (ISO 8601)"""
```

### WriteResult

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class WriteResult:
    """쓰기 작업 결과"""

    error: str | None = None
    """에러 메시지 (성공 시 None)"""

    path: str | None = None
    """생성된 파일 경로"""

    files_update: dict[str, Any] | None = None
    """상태 업데이트 (StateBackend용)"""
```

### EditResult

```python
@dataclass
class EditResult:
    """편집 작업 결과"""

    error: str | None = None
    """에러 메시지 (성공 시 None)"""

    path: str | None = None
    """편집된 파일 경로"""

    files_update: dict[str, Any] | None = None
    """상태 업데이트 (StateBackend용)"""

    occurrences: int | None = None
    """교체된 발생 횟수"""
```

### GrepMatch

```python
class GrepMatch(TypedDict):
    """검색 결과 매치"""

    path: str
    """매칭된 파일 경로"""

    line: int
    """라인 번호 (1-indexed)"""

    text: str
    """매칭 라인 내용"""
```

---

## SandboxBackendProtocol

명령 실행을 지원하는 확장 프로토콜입니다.

```python
class SandboxBackendProtocol(BackendProtocol):
    """격리된 실행 환경을 제공하는 확장 프로토콜"""

    @abc.abstractmethod
    def execute(self, command: str) -> "ExecuteResponse":
        """셸 명령 실행"""

    @property
    @abc.abstractmethod
    def id(self) -> str:
        """샌드박스 고유 식별자"""

@dataclass
class ExecuteResponse:
    """명령 실행 결과"""

    output: str
    """stdout + stderr 출력"""

    exit_code: int | None
    """종료 코드 (0: 성공)"""

    truncated: bool = False
    """출력 잘림 여부"""
```

---

## 기본 구현 예시: 메모리 백엔드

```python
from datetime import datetime, timezone
from deepagents.backends.protocol import (
    BackendProtocol,
    FileInfo,
    WriteResult,
    EditResult,
    GrepMatch,
    FileUploadResponse,
    FileDownloadResponse,
)

class InMemoryBackend(BackendProtocol):
    """인메모리 파일 시스템 백엔드

    파일을 메모리에 저장합니다. 테스트나 임시 작업에 적합합니다.
    """

    def __init__(self):
        """백엔드 초기화"""
        # 경로 -> 파일 내용 매핑
        self._files: dict[str, str] = {}
        # 경로 -> 메타데이터 매핑
        self._metadata: dict[str, dict] = {}

    def _get_timestamp(self) -> str:
        """현재 시각을 ISO 8601 형식으로 반환"""
        return datetime.now(timezone.utc).isoformat()

    def ls_info(self, path: str) -> list[FileInfo]:
        """디렉토리 목록 조회

        Args:
            path: 디렉토리 경로

        Returns:
            디렉토리 내 파일/폴더 정보 목록
        """
        # 경로 정규화
        if not path.endswith("/"):
            path = path + "/"

        infos: list[FileInfo] = []
        seen_dirs: set[str] = set()

        for file_path in self._files:
            if not file_path.startswith(path):
                continue

            # path 이후의 상대 경로
            relative = file_path[len(path):]

            if "/" in relative:
                # 하위 디렉토리
                subdir = relative.split("/")[0]
                dir_path = path + subdir + "/"
                if dir_path not in seen_dirs:
                    seen_dirs.add(dir_path)
                    infos.append({
                        "path": dir_path,
                        "is_dir": True,
                        "size": 0,
                        "modified_at": "",
                    })
            else:
                # 직접 파일
                meta = self._metadata.get(file_path, {})
                infos.append({
                    "path": file_path,
                    "is_dir": False,
                    "size": len(self._files[file_path]),
                    "modified_at": meta.get("modified_at", ""),
                })

        return sorted(infos, key=lambda x: x["path"])

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """파일 읽기

        Args:
            file_path: 파일 경로
            offset: 시작 라인 (0-indexed)
            limit: 읽을 최대 라인 수

        Returns:
            라인 번호가 포함된 파일 내용
        """
        if file_path not in self._files:
            return f"Error: File '{file_path}' not found"

        content = self._files[file_path]
        lines = content.splitlines()

        # 페이지네이션 적용
        selected = lines[offset:offset + limit]

        # cat -n 형식으로 포맷팅
        result_lines = []
        for i, line in enumerate(selected, start=offset + 1):
            result_lines.append(f"{i:6}\t{line}")

        return "\n".join(result_lines)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """새 파일 생성

        Args:
            file_path: 파일 경로
            content: 파일 내용

        Returns:
            쓰기 결과
        """
        if file_path in self._files:
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists."
            )

        self._files[file_path] = content
        self._metadata[file_path] = {
            "created_at": self._get_timestamp(),
            "modified_at": self._get_timestamp(),
        }

        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """파일 편집

        Args:
            file_path: 파일 경로
            old_string: 교체할 문자열
            new_string: 새 문자열
            replace_all: 모든 발생 교체 여부

        Returns:
            편집 결과
        """
        if file_path not in self._files:
            return EditResult(error=f"Error: File '{file_path}' not found")

        content = self._files[file_path]

        # 발생 횟수 확인
        count = content.count(old_string)

        if count == 0:
            return EditResult(error=f"old_string not found in {file_path}")

        if count > 1 and not replace_all:
            return EditResult(
                error=f"old_string found {count} times. "
                      f"Use replace_all=True or provide more context."
            )

        # 교체 수행
        if replace_all:
            new_content = content.replace(old_string, new_string)
            occurrences = count
        else:
            new_content = content.replace(old_string, new_string, 1)
            occurrences = 1

        self._files[file_path] = new_content
        self._metadata[file_path]["modified_at"] = self._get_timestamp()

        return EditResult(path=file_path, occurrences=occurrences)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """패턴 검색

        Args:
            pattern: 검색할 문자열
            path: 검색 디렉토리
            glob: 파일 필터 패턴

        Returns:
            매칭 결과 목록 또는 에러 메시지
        """
        import fnmatch

        matches: list[GrepMatch] = []
        search_path = path or "/"

        for file_path, content in self._files.items():
            # 경로 필터링
            if not file_path.startswith(search_path):
                continue

            # 글롭 필터링
            if glob and not fnmatch.fnmatch(file_path, glob):
                continue

            # 라인별 검색
            for i, line in enumerate(content.splitlines(), start=1):
                if pattern in line:
                    matches.append({
                        "path": file_path,
                        "line": i,
                        "text": line,
                    })

        return matches

    def glob_info(
        self,
        pattern: str,
        path: str = "/",
    ) -> list[FileInfo]:
        """글롭 패턴 검색

        Args:
            pattern: 글롭 패턴
            path: 검색 시작 디렉토리

        Returns:
            매칭된 파일 정보 목록
        """
        import fnmatch

        infos: list[FileInfo] = []

        for file_path in self._files:
            if not file_path.startswith(path):
                continue

            # 패턴 매칭
            if fnmatch.fnmatch(file_path, pattern):
                meta = self._metadata.get(file_path, {})
                infos.append({
                    "path": file_path,
                    "is_dir": False,
                    "size": len(self._files[file_path]),
                    "modified_at": meta.get("modified_at", ""),
                })

        return infos

    def upload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        """여러 파일 업로드"""
        responses = []
        for path, content in files:
            try:
                self._files[path] = content.decode("utf-8")
                self._metadata[path] = {
                    "created_at": self._get_timestamp(),
                    "modified_at": self._get_timestamp(),
                }
                responses.append(FileUploadResponse(path=path, error=None))
            except Exception as e:
                responses.append(FileUploadResponse(path=path, error=str(e)))
        return responses

    def download_files(
        self,
        paths: list[str],
    ) -> list[FileDownloadResponse]:
        """여러 파일 다운로드"""
        responses = []
        for path in paths:
            if path in self._files:
                content = self._files[path].encode("utf-8")
                responses.append(FileDownloadResponse(
                    path=path,
                    content=content,
                    error=None,
                ))
            else:
                responses.append(FileDownloadResponse(
                    path=path,
                    content=None,
                    error="file_not_found",
                ))
        return responses
```

---

## 실행 지원 백엔드 예시: Docker 백엔드

```python
import subprocess
import uuid
from deepagents.backends.protocol import (
    SandboxBackendProtocol,
    ExecuteResponse,
)

class DockerBackend(InMemoryBackend, SandboxBackendProtocol):
    """Docker 컨테이너 기반 백엔드

    파일은 메모리에 저장하고, 명령은 Docker 컨테이너에서 실행합니다.
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: float = 120.0,
        max_output_bytes: int = 100_000,
    ):
        """Docker 백엔드 초기화

        Args:
            image: 사용할 Docker 이미지
            timeout: 명령 실행 타임아웃(초)
            max_output_bytes: 최대 출력 크기
        """
        super().__init__()
        self._image = image
        self._timeout = timeout
        self._max_output_bytes = max_output_bytes
        self._container_id: str | None = None
        self._sandbox_id = f"docker-{uuid.uuid4().hex[:8]}"

    @property
    def id(self) -> str:
        """샌드박스 ID"""
        return self._sandbox_id

    def _ensure_container(self):
        """컨테이너가 실행 중인지 확인하고 필요시 시작"""
        if self._container_id is None:
            # 컨테이너 시작
            result = subprocess.run(
                [
                    "docker", "run", "-d",
                    "--rm",
                    self._image,
                    "tail", "-f", "/dev/null",  # 컨테이너 유지
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"컨테이너 시작 실패: {result.stderr}")
            self._container_id = result.stdout.strip()

    def execute(self, command: str) -> ExecuteResponse:
        """Docker 컨테이너에서 명령 실행

        Args:
            command: 실행할 셸 명령

        Returns:
            실행 결과
        """
        self._ensure_container()

        try:
            result = subprocess.run(
                [
                    "docker", "exec",
                    self._container_id,
                    "sh", "-c", command,
                ],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )

            # stdout과 stderr 결합
            output = result.stdout
            if result.stderr:
                stderr_lines = result.stderr.strip().split("\n")
                output += "\n" + "\n".join(
                    f"[stderr] {line}" for line in stderr_lines
                )

            # 출력 크기 제한
            truncated = False
            if len(output) > self._max_output_bytes:
                output = output[:self._max_output_bytes]
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
                truncated = True

            return ExecuteResponse(
                output=output or "<no output>",
                exit_code=result.returncode,
                truncated=truncated,
            )

        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output=f"Error: Command timed out after {self._timeout} seconds.",
                exit_code=124,
                truncated=False,
            )
        except Exception as e:
            return ExecuteResponse(
                output=f"Error executing command: {e}",
                exit_code=1,
                truncated=False,
            )

    def __del__(self):
        """정리: 컨테이너 중지"""
        if self._container_id:
            subprocess.run(
                ["docker", "stop", self._container_id],
                capture_output=True,
            )
```

---

## 비동기 메서드

모든 동기 메서드는 `a` 접두사가 붙은 비동기 버전을 가집니다.

```python
class AsyncBackend(BackendProtocol):
    """비동기 백엔드 예시"""

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """비동기 파일 읽기"""
        # 비동기 I/O 구현
        pass

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """비동기 파일 쓰기"""
        pass

    async def aexecute(self, command: str) -> ExecuteResponse:
        """비동기 명령 실행"""
        pass
```

기본 구현은 동기 메서드를 호출합니다:

```python
# BackendProtocol 기본 구현
async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
    return self.read(file_path, offset=offset, limit=limit)
```

---

## 클라우드 스토리지 백엔드 예시

```python
import boto3
from deepagents.backends.protocol import BackendProtocol, FileInfo, WriteResult

class S3Backend(BackendProtocol):
    """AWS S3 백엔드

    S3 버킷을 파일 시스템으로 사용합니다.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str = "us-east-1",
    ):
        """S3 백엔드 초기화

        Args:
            bucket: S3 버킷 이름
            prefix: 키 접두사 (가상 디렉토리)
            region: AWS 리전
        """
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._client = boto3.client("s3", region_name=region)

    def _to_s3_key(self, path: str) -> str:
        """파일 경로를 S3 키로 변환"""
        # /로 시작하면 제거
        if path.startswith("/"):
            path = path[1:]
        if self._prefix:
            return f"{self._prefix}/{path}"
        return path

    def _from_s3_key(self, key: str) -> str:
        """S3 키를 파일 경로로 변환"""
        if self._prefix and key.startswith(self._prefix + "/"):
            key = key[len(self._prefix) + 1:]
        return "/" + key

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """S3에서 파일 읽기"""
        key = self._to_s3_key(file_path)

        try:
            response = self._client.get_object(
                Bucket=self._bucket,
                Key=key,
            )
            content = response["Body"].read().decode("utf-8")

            # 페이지네이션
            lines = content.splitlines()
            selected = lines[offset:offset + limit]

            # cat -n 형식
            result_lines = []
            for i, line in enumerate(selected, start=offset + 1):
                result_lines.append(f"{i:6}\t{line}")
            return "\n".join(result_lines)

        except self._client.exceptions.NoSuchKey:
            return f"Error: File '{file_path}' not found"

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """S3에 파일 쓰기"""
        key = self._to_s3_key(file_path)

        # 이미 존재하는지 확인
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists."
            )
        except self._client.exceptions.ClientError:
            pass  # 존재하지 않음 - OK

        # 업로드
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=content.encode("utf-8"),
        )

        return WriteResult(path=file_path)

    def ls_info(self, path: str) -> list[FileInfo]:
        """S3 객체 목록 조회"""
        prefix = self._to_s3_key(path)
        if not prefix.endswith("/"):
            prefix = prefix + "/"

        paginator = self._client.get_paginator("list_objects_v2")
        infos: list[FileInfo] = []
        seen_dirs: set[str] = set()

        for page in paginator.paginate(
            Bucket=self._bucket,
            Prefix=prefix,
            Delimiter="/",
        ):
            # 디렉토리 (공통 접두사)
            for common_prefix in page.get("CommonPrefixes", []):
                dir_path = self._from_s3_key(common_prefix["Prefix"])
                if dir_path not in seen_dirs:
                    seen_dirs.add(dir_path)
                    infos.append({
                        "path": dir_path,
                        "is_dir": True,
                        "size": 0,
                    })

            # 파일
            for obj in page.get("Contents", []):
                file_path = self._from_s3_key(obj["Key"])
                if file_path.endswith("/"):
                    continue  # 디렉토리 마커 스킵
                infos.append({
                    "path": file_path,
                    "is_dir": False,
                    "size": obj["Size"],
                    "modified_at": obj["LastModified"].isoformat(),
                })

        return sorted(infos, key=lambda x: x["path"])

    # edit, grep_raw, glob_info 등도 구현 필요...
```

---

## 백엔드 등록

### create_deep_agent에서 등록

```python
from deepagents import create_deep_agent

# 직접 인스턴스
backend = InMemoryBackend()
agent = create_deep_agent(backend=backend)

# 팩토리 함수 (런타임에 생성)
agent = create_deep_agent(
    backend=lambda rt: InMemoryBackend()
)
```

### CompositeBackend로 조합

```python
from deepagents.backends.composite import CompositeBackend
from deepagents.backends import StateBackend

# 여러 백엔드 조합
composite = CompositeBackend(
    default=StateBackend(),  # 기본 백엔드
    routes={
        "/s3/": S3Backend(bucket="my-bucket"),
        "/docker/": DockerBackend(),
    },
)

agent = create_deep_agent(backend=composite)
```

---

## 테스트

```python
import pytest

def test_memory_backend_read_write():
    """읽기/쓰기 기본 테스트"""
    backend = InMemoryBackend()

    # 쓰기
    result = backend.write("/test.txt", "Hello\nWorld")
    assert result.error is None
    assert result.path == "/test.txt"

    # 읽기
    content = backend.read("/test.txt")
    assert "Hello" in content
    assert "World" in content

def test_memory_backend_edit():
    """편집 테스트"""
    backend = InMemoryBackend()
    backend.write("/test.txt", "Hello World")

    # 편집
    result = backend.edit("/test.txt", "World", "Universe")
    assert result.error is None
    assert result.occurrences == 1

    # 확인
    content = backend.read("/test.txt")
    assert "Universe" in content

def test_docker_backend_execute():
    """명령 실행 테스트"""
    backend = DockerBackend()

    result = backend.execute("echo 'Hello'")
    assert result.exit_code == 0
    assert "Hello" in result.output
```

---

## 모범 사례

### 1. 에러 처리

```python
def read(self, file_path: str, ...) -> str:
    try:
        # 구현
        pass
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found"
    except PermissionError:
        return f"Error: Permission denied for '{file_path}'"
    except Exception as e:
        return f"Error: {e}"
```

### 2. 경로 정규화

```python
def _normalize_path(self, path: str) -> str:
    """경로 정규화"""
    import os
    # 정규화
    path = os.path.normpath(path)
    # 슬래시 통일
    path = path.replace("\\", "/")
    # 선행 슬래시 보장
    if not path.startswith("/"):
        path = "/" + path
    return path
```

### 3. 상태 업데이트 (StateBackend 호환)

```python
def write(self, file_path: str, content: str) -> WriteResult:
    # 실제 쓰기...

    # StateBackend와 호환되려면 files_update 반환
    return WriteResult(
        path=file_path,
        files_update={
            file_path: {
                "content": content.splitlines(),
                "created_at": self._get_timestamp(),
                "modified_at": self._get_timestamp(),
            }
        },
    )
```

---

## 관련 문서

- [백엔드 시스템](../01-architecture/backend-system.md)
- [백엔드 API](../05-api-reference/backend-api.md)
- [create_deep_agent API](../05-api-reference/create-deep-agent-api.md)
