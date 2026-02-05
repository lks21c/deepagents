# 백엔드 시스템 (Backend System)

> 다양한 저장소와 실행 환경을 추상화하는 플러그형 백엔드 아키텍처를 설명합니다.

## 개요

백엔드 시스템은 Deep Agents가 다양한 환경(로컬 파일시스템, 인메모리 상태, 원격 저장소, 샌드박스)에서 일관되게 동작하도록 추상화 계층을 제공합니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Backend Architecture                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                    ┌──────────────────────┐                         │
│                    │   Deep Agent         │                         │
│                    │   (Middleware)       │                         │
│                    └──────────┬───────────┘                         │
│                               │                                      │
│                    ┌──────────▼───────────┐                         │
│                    │   BackendProtocol    │                         │
│                    │   (Abstract API)     │                         │
│                    └──────────┬───────────┘                         │
│                               │                                      │
│           ┌───────────────────┼───────────────────┐                 │
│           │                   │                   │                 │
│           ▼                   ▼                   ▼                 │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │
│   │ Filesystem   │   │ State        │   │ Sandbox      │           │
│   │ Backend      │   │ Backend      │   │ Backend      │           │
│   └──────────────┘   └──────────────┘   └──────────────┘           │
│           │                   │                   │                 │
│           ▼                   ▼                   ▼                 │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │
│   │ 로컬 디스크   │   │ LangGraph    │   │ 격리 컨테이너 │           │
│   │              │   │ State        │   │              │           │
│   └──────────────┘   └──────────────┘   └──────────────┘           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## BackendProtocol

**소스 위치**: `libs/deepagents/deepagents/backends/protocol.py:198-457`

모든 백엔드가 구현해야 하는 추상 기본 클래스입니다.

### 핵심 인터페이스

```python
class BackendProtocol(abc.ABC):
    """플러그형 메모리 백엔드를 위한 프로토콜 추상 기본 클래스"""

    def ls_info(self, path: str) -> list["FileInfo"]:
        """디렉토리 내 파일 목록 조회"""

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """파일 내용 읽기 (페이지네이션 지원)"""

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """새 파일 생성"""

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """기존 파일 편집"""

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list["GrepMatch"] | str:
        """텍스트 패턴 검색"""

    def glob_info(self, pattern: str, path: str = "/") -> list["FileInfo"]:
        """글롭 패턴으로 파일 찾기"""

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """여러 파일 업로드"""

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """여러 파일 다운로드"""
```

### 비동기 메서드

모든 동기 메서드는 `a` 접두사가 붙은 비동기 버전을 가집니다:

```python
# 동기
content = backend.read("/path/to/file.py")

# 비동기
content = await backend.aread("/path/to/file.py")
```

---

## 데이터 타입

### FileInfo

**소스 위치**: `protocol.py:107-124`

파일 메타데이터를 담는 TypedDict입니다.

```python
class FileInfo(TypedDict):
    """구조화된 파일 목록 정보"""

    path: str                           # 파일의 절대 경로 (필수)
    is_dir: NotRequired[bool]           # 디렉토리 여부 (선택)
    size: NotRequired[int]              # 파일 크기(바이트) (선택)
    modified_at: NotRequired[str]       # ISO 타임스탬프 (선택)
```

**설계 이유**:
- `path`만 필수: 최소한의 계약으로 다양한 백엔드 지원
- 선택적 필드: 백엔드에 따라 정보 제공 수준이 다름

### GrepMatch

**소스 위치**: `protocol.py:127-142`

검색 결과 항목을 담는 TypedDict입니다.

```python
class GrepMatch(TypedDict):
    """구조화된 grep 검색 결과 항목"""

    path: str   # 매칭 파일의 절대 경로
    line: int   # 라인 번호 (1부터 시작)
    text: str   # 매칭된 라인의 전체 텍스트
```

### WriteResult / EditResult

**소스 위치**: `protocol.py:144-196`

파일 작업 결과를 담는 데이터클래스입니다.

```python
@dataclass
class WriteResult:
    """백엔드 쓰기 작업의 결과"""

    error: str | None = None              # 오류 메시지, 성공 시 None
    path: str | None = None               # 작성된 파일의 절대 경로
    files_update: dict[str, Any] | None = None  # 상태 업데이트 딕셔너리

@dataclass
class EditResult:
    """백엔드 편집 작업의 결과"""

    error: str | None = None
    path: str | None = None
    files_update: dict[str, Any] | None = None
    occurrences: int | None = None        # 교체 횟수
```

**`files_update` 필드 설명**:

| 백엔드 유형 | `files_update` 값 | 이유 |
|------------|-------------------|------|
| StateBackend | `{file_path: file_data}` | LangGraph 상태에 저장 |
| FilesystemBackend | `None` | 이미 디스크에 저장됨 |
| 원격 백엔드 | `None` | 이미 원격 저장소에 저장됨 |

---

## FileOperationError

**소스 위치**: `protocol.py:35-50`

파일 작업 오류를 표준화하는 리터럴 타입입니다.

```python
FileOperationError = Literal[
    "file_not_found",      # 파일이 존재하지 않음
    "permission_denied",   # 접근 거부됨
    "is_directory",        # 디렉토리를 파일로 다운로드 시도
    "invalid_path",        # 경로 문법 오류
]
```

**설계 이유**:
- LLM이 이해하고 복구할 수 있는 구조화된 오류
- 예외 대신 반환값으로 오류 전달 → 배치 작업에서 부분 성공 허용

---

## SandboxBackendProtocol

**소스 위치**: `protocol.py:482-525`

격리된 실행 환경을 제공하는 확장 프로토콜입니다.

```python
class SandboxBackendProtocol(BackendProtocol):
    """격리된 런타임을 가진 샌드박스 백엔드"""

    def execute(self, command: str) -> ExecuteResponse:
        """샌드박스 내에서 셸 명령 실행"""

    @property
    def id(self) -> str:
        """샌드박스 인스턴스의 고유 식별자"""
```

### ExecuteResponse

```python
@dataclass
class ExecuteResponse:
    """코드 실행 결과"""

    output: str              # stdout + stderr 결합 출력
    exit_code: int | None    # 종료 코드 (0=성공)
    truncated: bool = False  # 출력이 잘렸는지 여부
```

---

## 백엔드 구현체

### FilesystemBackend

**소스 위치**: `libs/deepagents/deepagents/backends/local_shell.py`

로컬 파일시스템에 접근하는 백엔드입니다.

```python
from deepagents.backends import FilesystemBackend

# 루트 디렉토리 제한
backend = FilesystemBackend(root_dir="/workspace")

# 파일 읽기
content = backend.read("/workspace/src/main.py")

# 파일 목록
files = backend.ls_info("/workspace/src")

# 패턴 검색
matches = backend.grep_raw("def ", path="/workspace/src", glob="*.py")
```

**보안 고려사항**:
- `root_dir` 외부 접근 차단
- 경로 탐색(path traversal) 공격 방지
- 심볼릭 링크 탈출 방지

### StateBackend

**소스 위치**: `libs/deepagents/deepagents/backends/state.py`

LangGraph 상태에 파일을 저장하는 인메모리 백엔드입니다.

```python
from deepagents.backends import StateBackend

# 런타임에서 상태 접근 (팩토리 패턴 사용)
def get_state_backend(runtime):
    return StateBackend(runtime)

agent = create_deep_agent(
    backend=get_state_backend,  # 팩토리 함수 전달
)
```

**사용 사례**:
- 테스트 환경
- 임시 파일 작업
- 서버리스 함수

**파일 데이터 구조**:

```python
{
    "content": list[str],      # 텍스트 콘텐츠 라인들
    "created_at": str,         # ISO 형식 타임스탬프
    "modified_at": str,        # ISO 형식 타임스탬프
}
```

---

## BackendFactory 패턴

**소스 위치**: `protocol.py:527-531`

런타임에 백엔드를 생성하는 팩토리 함수 타입입니다.

```python
BackendFactory: TypeAlias = Callable[[ToolRuntime], BackendProtocol]
```

**사용 이유**:
- StateBackend는 런타임 상태에 접근해야 함
- 팩토리를 통해 지연 초기화(lazy initialization) 가능

```python
# 직접 인스턴스 전달 (FilesystemBackend)
agent = create_deep_agent(
    backend=FilesystemBackend(root_dir="/workspace"),
)

# 팩토리 함수 전달 (StateBackend)
agent = create_deep_agent(
    backend=lambda runtime: StateBackend(runtime),
)
```

---

## 백엔드와 미들웨어 통합

### 미들웨어에서 백엔드 사용

```python
class MemoryMiddleware(AgentMiddleware):
    def __init__(self, *, backend: BACKEND_TYPES, sources: list[str]):
        self._backend = backend
        self.sources = sources

    def _get_backend(self, state, runtime, config) -> BackendProtocol:
        """인스턴스 또는 팩토리에서 백엔드 해석"""
        if callable(self._backend):
            # 팩토리 함수인 경우 런타임으로 호출
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            return self._backend(tool_runtime)
        # 직접 인스턴스인 경우 그대로 반환
        return self._backend
```

**설계 이유**:
- 두 가지 백엔드 유형 모두 지원
- 런타임 의존성이 필요한 백엔드에 유연성 제공

---

## 커스텀 백엔드 구현

### 기본 구현 예제

```python
from deepagents.backends.protocol import (
    BackendProtocol,
    FileInfo,
    GrepMatch,
    WriteResult,
    EditResult,
)

class S3Backend(BackendProtocol):
    """Amazon S3를 저장소로 사용하는 백엔드"""

    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix
        self.s3 = boto3.client("s3")

    def ls_info(self, path: str) -> list[FileInfo]:
        """S3 버킷에서 객체 목록 조회"""
        response = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self._resolve_path(path),
        )
        return [
            FileInfo(
                path=obj["Key"],
                size=obj["Size"],
                modified_at=obj["LastModified"].isoformat(),
            )
            for obj in response.get("Contents", [])
        ]

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """S3에서 객체 내용 읽기"""
        response = self.s3.get_object(
            Bucket=self.bucket,
            Key=self._resolve_path(file_path),
        )
        content = response["Body"].read().decode("utf-8")
        lines = content.split("\n")
        return self._format_with_line_numbers(lines[offset:offset + limit])

    def write(self, file_path: str, content: str) -> WriteResult:
        """S3에 새 객체 생성"""
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=self._resolve_path(file_path),
                Body=content.encode("utf-8"),
            )
            return WriteResult(path=file_path)
        except Exception as e:
            return WriteResult(error=str(e))

    # ... 나머지 메서드 구현
```

---

## 사용 예제

### 로컬 개발 환경

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    backend=FilesystemBackend(root_dir="./project"),
    memory=["./AGENTS.md"],
    skills=["./skills/"],
)
```

### 테스트 환경

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend

agent = create_deep_agent(
    backend=lambda rt: StateBackend(rt),
)

# 테스트용 파일 업로드
agent.backend.upload_files([
    ("/test/config.json", b'{"key": "value"}'),
])
```

### 프로덕션 환경 (샌드박스)

```python
from deepagents import create_deep_agent
from my_company.backends import KubernetesSandbox

sandbox = KubernetesSandbox(
    image="python:3.11",
    resource_limits={"cpu": "1", "memory": "512Mi"},
)

agent = create_deep_agent(
    backend=sandbox,
)

# 코드 실행
result = sandbox.execute("python script.py")
print(result.output)
```

---

## 다음 단계

- [상태 관리](./state-management.md)
- [미들웨어 시스템](./middleware-system.md)
- [커스텀 백엔드 구현](../06-advanced/custom-backends.md)
