# 백엔드 API 레퍼런스

> Deep Agents 백엔드 프로토콜 및 구현체의 API 레퍼런스입니다.

## BackendProtocol

**소스 위치**: `libs/deepagents/deepagents/backends/protocol.py:198-457`

모든 백엔드가 구현해야 하는 추상 기본 클래스입니다.

```python
class BackendProtocol(abc.ABC):
    """플러그형 메모리 백엔드를 위한 프로토콜"""

    def ls_info(self, path: str) -> list[FileInfo]: ...
    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str: ...
    def write(self, file_path: str, content: str) -> WriteResult: ...
    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult: ...
    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch] | str: ...
    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]: ...
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]: ...
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]: ...
```

---

## 메서드

### ls_info

디렉토리 내 파일 목록을 조회합니다.

```python
def ls_info(self, path: str) -> list[FileInfo]
```

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `path` | `str` | 디렉토리의 절대 경로 |

**반환값**: `list[FileInfo]` - 파일 메타데이터 목록

---

### read

파일 내용을 읽습니다.

```python
def read(
    self,
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
) -> str
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| `file_path` | `str` | 필수 | 파일의 절대 경로 |
| `offset` | `int` | 0 | 시작 라인 (0-indexed) |
| `limit` | `int` | 2000 | 읽을 최대 라인 수 |

**반환값**: `str` - 라인 번호가 포함된 파일 내용 (cat -n 형식)

---

### write

새 파일을 생성합니다.

```python
def write(
    self,
    file_path: str,
    content: str,
) -> WriteResult
```

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `file_path` | `str` | 파일의 절대 경로 |
| `content` | `str` | 파일 내용 |

**반환값**: `WriteResult` - 쓰기 작업 결과

---

### edit

기존 파일을 편집합니다.

```python
def edit(
    self,
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> EditResult
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| `file_path` | `str` | 필수 | 파일의 절대 경로 |
| `old_string` | `str` | 필수 | 교체할 문자열 |
| `new_string` | `str` | 필수 | 새 문자열 |
| `replace_all` | `bool` | False | 모든 발생 교체 여부 |

**반환값**: `EditResult` - 편집 작업 결과

---

### grep_raw

텍스트 패턴을 검색합니다.

```python
def grep_raw(
    self,
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
) -> list[GrepMatch] | str
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| `pattern` | `str` | 필수 | 검색할 리터럴 문자열 |
| `path` | `str \| None` | None | 검색 디렉토리 |
| `glob` | `str \| None` | None | 파일 필터 패턴 |

**반환값**: `list[GrepMatch]` (성공) 또는 `str` (오류 메시지)

---

### glob_info

글롭 패턴으로 파일을 찾습니다.

```python
def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| `pattern` | `str` | 필수 | 글롭 패턴 |
| `path` | `str` | "/" | 검색 시작 디렉토리 |

**반환값**: `list[FileInfo]` - 매칭된 파일 목록

---

### upload_files / download_files

여러 파일을 업로드/다운로드합니다.

```python
def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]
def download_files(self, paths: list[str]) -> list[FileDownloadResponse]
```

---

## 데이터 타입

### FileInfo

```python
class FileInfo(TypedDict):
    path: str                           # 필수
    is_dir: NotRequired[bool]           # 선택
    size: NotRequired[int]              # 선택
    modified_at: NotRequired[str]       # 선택
```

### GrepMatch

```python
class GrepMatch(TypedDict):
    path: str   # 파일 경로
    line: int   # 라인 번호 (1-indexed)
    text: str   # 매칭 라인 내용
```

### WriteResult

```python
@dataclass
class WriteResult:
    error: str | None = None
    path: str | None = None
    files_update: dict[str, Any] | None = None
```

### EditResult

```python
@dataclass
class EditResult:
    error: str | None = None
    path: str | None = None
    files_update: dict[str, Any] | None = None
    occurrences: int | None = None
```

### FileOperationError

```python
FileOperationError = Literal[
    "file_not_found",
    "permission_denied",
    "is_directory",
    "invalid_path",
]
```

---

## SandboxBackendProtocol

격리된 실행 환경을 제공하는 확장 프로토콜입니다.

```python
class SandboxBackendProtocol(BackendProtocol):

    def execute(self, command: str) -> ExecuteResponse:
        """셸 명령 실행"""

    @property
    def id(self) -> str:
        """샌드박스 ID"""
```

### ExecuteResponse

```python
@dataclass
class ExecuteResponse:
    output: str              # stdout + stderr
    exit_code: int | None    # 종료 코드
    truncated: bool = False  # 출력 잘림 여부
```

---

## 구현체

### FilesystemBackend

로컬 파일시스템 백엔드입니다.

```python
from deepagents.backends import FilesystemBackend

backend = FilesystemBackend(root_dir="./workspace")
```

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `root_dir` | `str` | 루트 디렉토리 (이 외부 접근 차단) |

---

### StateBackend

LangGraph 상태 기반 인메모리 백엔드입니다.

```python
from deepagents.backends import StateBackend

# 팩토리 패턴으로 사용
backend_factory = lambda rt: StateBackend(rt)
```

---

## BackendFactory

런타임에 백엔드를 생성하는 팩토리 함수 타입입니다.

```python
BackendFactory: TypeAlias = Callable[[ToolRuntime], BackendProtocol]
```

**사용 예시**:

```python
# FilesystemBackend: 직접 인스턴스
agent = create_deep_agent(
    backend=FilesystemBackend(root_dir="./"),
)

# StateBackend: 팩토리 함수
agent = create_deep_agent(
    backend=lambda rt: StateBackend(rt),
)
```

---

## 비동기 메서드

모든 동기 메서드는 `a` 접두사가 붙은 비동기 버전을 가집니다:

| 동기 | 비동기 |
|------|--------|
| `ls_info` | `als_info` |
| `read` | `aread` |
| `write` | `awrite` |
| `edit` | `aedit` |
| `grep_raw` | `agrep_raw` |
| `glob_info` | `aglob_info` |
| `upload_files` | `aupload_files` |
| `download_files` | `adownload_files` |
| `execute` | `aexecute` |

---

## 관련 문서

- [백엔드 시스템](../01-architecture/backend-system.md)
- [커스텀 백엔드 구현](../06-advanced/custom-backends.md)
- [create_deep_agent API](./create-deep-agent-api.md)
