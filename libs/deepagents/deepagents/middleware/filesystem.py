"""
모듈명: filesystem.py
설명: 에이전트에 파일시스템 도구를 제공하는 미들웨어

이 미들웨어는 에이전트에 파일시스템 조작 도구들을 추가합니다:
    - ls: 디렉토리 내용 목록 조회
    - read_file: 파일 읽기 (페이지네이션 지원)
    - write_file: 새 파일 생성 및 쓰기
    - edit_file: 기존 파일의 문자열 교체 편집
    - glob: 패턴 매칭으로 파일 검색
    - grep: 텍스트 패턴으로 파일 내용 검색

백엔드가 SandboxBackendProtocol을 구현하면 execute 도구도 추가됩니다:
    - execute: 샌드박스 환경에서 셸 명령 실행

또한 큰 도구 결과가 토큰 임계값을 초과하면 파일시스템으로
자동 오프로드하여 컨텍스트 윈도우 포화를 방지합니다.

주요 클래스:
    - FilesystemMiddleware: 파일시스템 도구 미들웨어
    - FilesystemState: 파일시스템 상태 스키마
    - FileData: 파일 메타데이터 저장 구조

사용 예시:
    ```python
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.backends import StateBackend

    # 기본 사용 (에이전트 상태에 파일 저장)
    agent = create_agent(middleware=[FilesystemMiddleware()])

    # 샌드박스 백엔드로 실행 도구 활성화
    from my_sandbox import DockerSandboxBackend
    sandbox = DockerSandboxBackend(container_id="my-container")
    agent = create_agent(middleware=[FilesystemMiddleware(backend=sandbox)])
    ```
"""
# ruff: noqa: E501

import os
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Literal, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command
from typing_extensions import TypedDict

from deepagents.backends import StateBackend
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import (
    BACKEND_TYPES as BACKEND_TYPES,  # 하위 호환성을 위해 여기서 타입 재내보내기
    BackendProtocol,
    EditResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.utils import (
    format_content_with_line_numbers,
    format_grep_matches,
    sanitize_tool_call_id,
    truncate_if_too_long,
)
from deepagents.middleware._utils import append_to_system_message

# ============================================================================
# 상수 정의
# ============================================================================

# 빈 파일 읽기 시 표시되는 경고 메시지
EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"

# 라인 번호 표시 시 사용되는 너비 (자릿수)
LINE_NUMBER_WIDTH = 6

# read_file 도구의 기본 오프셋 (0부터 시작)
DEFAULT_READ_OFFSET = 0

# read_file 도구의 기본 읽기 라인 수 제한
DEFAULT_READ_LIMIT = 100

# read_file에서 크기 제한으로 잘릴 때 표시되는 메시지 템플릿
# {file_path}는 런타임에 실제 경로로 치환됨
READ_FILE_TRUNCATION_MSG = (
    "\n\n[Output was truncated due to size limits. "
    "The file content is very large. "
    "Consider reformatting the file to make it easier to navigate. "
    "For example, if this is JSON, use execute(command='jq . {file_path}') to pretty-print it with line breaks. "
    "For other formats, you can use appropriate formatting tools to split long lines.]"
)

# 토큰당 대략적인 문자 수 (잘림 계산용)
# 보수적으로 토큰당 4자로 추정 (실제 비율은 콘텐츠에 따라 다름)
# 맞을 수 있는 콘텐츠의 조기 제거를 방지하기 위해 높은 쪽으로 설정
NUM_CHARS_PER_TOKEN = 4


class FileData(TypedDict):
    """파일 내용과 메타데이터를 저장하는 데이터 구조.

    에이전트 상태에 파일을 저장할 때 사용되는 TypedDict입니다.
    StateBackend에서 파일 정보를 관리하는 데 활용됩니다.

    Attributes:
        content: 파일의 각 라인을 담은 문자열 리스트.
        created_at: 파일 생성 시각 (ISO 8601 형식).
        modified_at: 마지막 수정 시각 (ISO 8601 형식).
    """

    content: list[str]
    """파일의 각 라인 목록."""

    created_at: str
    """파일 생성 시각 (ISO 8601 타임스탬프)."""

    modified_at: str
    """마지막 수정 시각 (ISO 8601 타임스탬프)."""


def _file_data_reducer(left: dict[str, FileData] | None, right: dict[str, FileData | None]) -> dict[str, FileData]:
    """파일 업데이트를 병합하며 삭제도 지원하는 리듀서.

    이 리듀서는 right 딕셔너리의 `None` 값을 삭제 마커로 처리하여
    파일 삭제를 가능하게 합니다. LangGraph의 상태 관리 시스템에서
    어노테이션된 리듀서가 상태 업데이트 병합을 제어하도록 설계되었습니다.

    Args:
        left: 기존 파일 딕셔너리. 초기화 시 `None`일 수 있음.
        right: 병합할 새 파일 딕셔너리. `None` 값을 가진 파일은
            삭제 마커로 처리되어 결과에서 제거됨.

    Returns:
        동일 키에 대해 right가 left를 덮어쓰고,
        right의 `None` 값은 삭제를 트리거하는 병합된 딕셔너리.

    사용 예시:
        ```python
        existing = {"/file1.txt": FileData(...), "/file2.txt": FileData(...)}
        updates = {"/file2.txt": None, "/file3.txt": FileData(...)}
        result = file_data_reducer(existing, updates)
        # 결과: {"/file1.txt": FileData(...), "/file3.txt": FileData(...)}
        # file2.txt는 삭제되고 file3.txt가 추가됨
        ```
    """
    # left가 None이면 right에서 None이 아닌 값만 반환
    if left is None:
        return {k: v for k, v in right.items() if v is not None}

    # left를 복사하여 시작
    result = {**left}

    # right의 각 항목을 처리
    for key, value in right.items():
        if value is None:
            # None 값은 삭제 마커로 처리
            result.pop(key, None)
        else:
            # None이 아니면 값을 덮어쓰기
            result[key] = value
    return result


def _validate_path(path: str, *, allowed_prefixes: Sequence[str] | None = None) -> str:
    r"""보안을 위해 파일 경로를 검증하고 정규화합니다.

    디렉토리 트래버설 공격을 방지하고 일관된 형식을 강제하여
    경로가 안전하게 사용될 수 있도록 보장합니다. 모든 경로는
    슬래시(/)를 사용하고 선행 슬래시로 시작하도록 정규화됩니다.

    이 함수는 가상 파일시스템 경로용으로 설계되었으며,
    일관성 유지와 경로 형식 모호성 방지를 위해
    Windows 절대 경로(예: C:/..., F:/...)를 거부합니다.

    Args:
        path: 검증하고 정규화할 경로.
        allowed_prefixes: 허용된 경로 접두사 목록 (선택사항).
            제공되면 정규화된 경로가 이 접두사 중 하나로 시작해야 함.

    Returns:
        `/`로 시작하고 슬래시를 사용하는 정규화된 정규 경로.

    Raises:
        ValueError: 경로에 트래버설 시퀀스(`..` 또는 `~`)가 포함되거나,
            Windows 절대 경로(예: C:/...)이거나,
            `allowed_prefixes`가 지정된 경우 허용된 접두사로 시작하지 않을 때.

    사용 예시:
        ```python
        _validate_path("foo/bar")  # 반환: "/foo/bar"
        _validate_path("/./foo//bar")  # 반환: "/foo/bar"
        _validate_path("../etc/passwd")  # ValueError 발생 (디렉토리 트래버설)
        _validate_path(r"C:\\Users\\file.txt")  # ValueError 발생 (Windows 경로)
        _validate_path("/data/file.txt", allowed_prefixes=["/data/"])  # 정상
        _validate_path("/etc/file.txt", allowed_prefixes=["/data/"])  # ValueError 발생
        ```
    """
    # 디렉토리 트래버설 시도 검사
    if ".." in path or path.startswith("~"):
        msg = f"Path traversal not allowed: {path}"
        raise ValueError(msg)

    # Windows 절대 경로 거부 (예: C:\..., D:/...)
    # 가상 파일시스템 경로의 일관성 유지를 위함
    if re.match(r"^[a-zA-Z]:", path):
        msg = f"Windows absolute paths are not supported: {path}. Please use virtual paths starting with / (e.g., /workspace/file.txt)"
        raise ValueError(msg)

    # 경로 정규화 수행
    normalized = os.path.normpath(path)
    # 백슬래시를 슬래시로 변환
    normalized = normalized.replace("\\", "/")

    # 선행 슬래시가 없으면 추가
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    # 허용된 접두사 검사 (지정된 경우)
    if allowed_prefixes is not None and not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        msg = f"Path must start with one of {allowed_prefixes}: {path}"
        raise ValueError(msg)

    return normalized


class FilesystemState(AgentState):
    """파일시스템 미들웨어의 상태 스키마.

    에이전트 상태에 포함되어 파일시스템 데이터를 추적합니다.
    _file_data_reducer를 통해 파일 추가, 수정, 삭제 작업이
    올바르게 병합됩니다.
    """

    files: Annotated[NotRequired[dict[str, FileData]], _file_data_reducer]
    """파일시스템 내의 파일 목록 (경로 -> FileData 매핑)."""


LIST_FILES_TOOL_DESCRIPTION = """Lists all files in a directory.

This is useful for exploring the filesystem and finding the right file to read or edit.
You should almost ALWAYS use this tool before using the read_file or edit_file tools."""

READ_FILE_TOOL_DESCRIPTION = """Reads a file from the filesystem.

Assume this tool is able to read all files. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- By default, it reads up to 100 lines starting from the beginning of the file
- **IMPORTANT for large files and codebase exploration**: Use pagination with offset and limit parameters to avoid context overflow
  - First scan: read_file(path, limit=100) to see file structure
  - Read more sections: read_file(path, offset=100, limit=200) for next 200 lines
  - Only omit limit (read full file) when necessary for editing
- Specify offset and limit: read_file(path, offset=0, limit=100) reads first 100 lines
- Results are returned using cat -n format, with line numbers starting at 1
- Lines longer than 5,000 characters will be split into multiple lines with continuation markers (e.g., 5.1, 5.2, etc.). When you specify a limit, these continuation lines count towards the limit.
- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
- You should ALWAYS make sure a file has been read before editing it."""

EDIT_FILE_TOOL_DESCRIPTION = """Performs exact string replacements in files.

Usage:
- You must read the file before editing. This tool will error if you attempt an edit without reading the file first.
- When editing, preserve the exact indentation (tabs/spaces) from the read output. Never include line number prefixes in old_string or new_string.
- ALWAYS prefer editing existing files over creating new ones.
- Only use emojis if the user explicitly requests it."""


WRITE_FILE_TOOL_DESCRIPTION = """Writes to a new file in the filesystem.

Usage:
- The write_file tool will create the a new file.
- Prefer to edit existing files (with the edit_file tool) over creating new ones when possible.
"""

GLOB_TOOL_DESCRIPTION = """Find files matching a glob pattern.

Supports standard glob patterns: `*` (any characters), `**` (any directories), `?` (single character).
Returns a list of absolute file paths that match the pattern.

Examples:
- `**/*.py` - Find all Python files
- `*.txt` - Find all text files in root
- `/subdir/**/*.md` - Find all markdown files under /subdir"""

GREP_TOOL_DESCRIPTION = """Search for a text pattern across files.

Searches for literal text (not regex) and returns matching files or content based on output_mode.
Special characters like parentheses, brackets, pipes, etc. are treated as literal characters, not regex operators.

Examples:
- Search all files: `grep(pattern="TODO")`
- Search Python files only: `grep(pattern="import", glob="*.py")`
- Show matching lines: `grep(pattern="error", output_mode="content")`
- Search for code with special chars: `grep(pattern="def __init__(self):")`"""

EXECUTE_TOOL_DESCRIPTION = """Executes a shell command in an isolated sandbox environment.

Usage:
Executes a given command in the sandbox environment with proper handling and security measures.
Before executing the command, please follow these steps:
1. Directory Verification:
   - If the command will create new directories or files, first use the ls tool to verify the parent directory exists and is the correct location
   - For example, before running "mkdir foo/bar", first use ls to check that "foo" exists and is the intended parent directory
2. Command Execution:
   - Always quote file paths that contain spaces with double quotes (e.g., cd "path with spaces/file.txt")
   - Examples of proper quoting:
     - cd "/Users/name/My Documents" (correct)
     - cd /Users/name/My Documents (incorrect - will fail)
     - python "/path/with spaces/script.py" (correct)
     - python /path/with spaces/script.py (incorrect - will fail)
   - After ensuring proper quoting, execute the command
   - Capture the output of the command
Usage notes:
  - Commands run in an isolated sandbox environment
  - Returns combined stdout/stderr output with exit code
  - If the output is very large, it may be truncated
  - VERY IMPORTANT: You MUST avoid using search commands like find and grep. Instead use the grep, glob tools to search. You MUST avoid read tools like cat, head, tail, and use read_file to read files.
  - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines (newlines are ok in quoted strings)
    - Use '&&' when commands depend on each other (e.g., "mkdir dir && cd dir")
    - Use ';' only when you need to run commands sequentially but don't care if earlier commands fail
  - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of cd

Examples:
  Good examples:
    - execute(command="pytest /foo/bar/tests")
    - execute(command="python /path/to/script.py")
    - execute(command="npm install && npm test")

  Bad examples (avoid these):
    - execute(command="cd /foo/bar && pytest tests")  # Use absolute path instead
    - execute(command="cat file.txt")  # Use read_file tool instead
    - execute(command="find . -name '*.py'")  # Use glob tool instead
    - execute(command="grep -r 'pattern' .")  # Use grep tool instead

Note: This tool is only available if the backend supports execution (SandboxBackendProtocol).
If execution is not supported, the tool will return an error message."""

FILESYSTEM_SYSTEM_PROMPT = """## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`

You have access to a filesystem which you can interact with using these tools.
All file paths must start with a /.

- ls: list files in a directory (requires absolute path)
- read_file: read a file from the filesystem
- write_file: write to a file in the filesystem
- edit_file: edit a file in the filesystem
- glob: find files matching a pattern (e.g., "**/*.py")
- grep: search for text within files"""

EXECUTION_SYSTEM_PROMPT = """## Execute Tool `execute`

You have access to an `execute` tool for running shell commands in a sandboxed environment.
Use this tool to run commands, scripts, tests, builds, and other shell operations.

- execute: run a shell command in the sandbox (returns output and exit code)"""


def _supports_execution(backend: BackendProtocol) -> bool:
    """백엔드가 명령 실행을 지원하는지 확인합니다.

    CompositeBackend의 경우 기본 백엔드가 실행을 지원하는지 확인하고,
    다른 백엔드의 경우 SandboxBackendProtocol 구현 여부를 확인합니다.

    Args:
        backend: 확인할 백엔드 인스턴스.

    Returns:
        백엔드가 실행을 지원하면 True, 그렇지 않으면 False.
    """
    # CompositeBackend인 경우 기본 백엔드 확인
    if isinstance(backend, CompositeBackend):
        return isinstance(backend.default, SandboxBackendProtocol)

    # 다른 백엔드는 isinstance 검사 사용
    return isinstance(backend, SandboxBackendProtocol)


# ============================================================================
# 대용량 결과 제거(eviction) 로직에서 제외할 도구 목록
# ============================================================================
#
# 이 튜플은 토큰 제한을 초과해도 결과가 파일시스템으로 제거되지 않아야 하는
# 도구들을 포함합니다. 도구들은 다양한 이유로 제외됩니다:
#
# 1. 내장 잘림(truncation) 기능이 있는 도구 (ls, glob, grep):
#    이 도구들은 출력이 너무 커지면 자체적으로 잘라냅니다. 많은 매치로 인해
#    잘린 출력이 생성되면, 이는 일반적으로 전체 결과 보존보다 쿼리 개선이
#    필요함을 나타냅니다. 이 경우 잘린 매치는 잠재적으로 노이즈에 가깝고
#    LLM은 검색 기준을 좁히도록 안내되어야 합니다.
#
# 2. 잘림 동작이 문제가 되는 도구 (read_file):
#    read_file은 처리하기 까다로운데, 실패 모드가 단일 긴 줄인 경우가 있기
#    때문입니다(예: 각 줄에 매우 긴 페이로드가 있는 jsonl 파일). read_file
#    결과를 잘라내면 에이전트가 read_file을 사용해 잘린 파일을 다시 읽으려
#    시도할 수 있는데, 이는 도움이 되지 않습니다.
#
# 3. 제한을 초과하지 않는 도구 (edit_file, write_file):
#    이 도구들은 최소한의 확인 메시지만 반환하며 토큰 제한을 초과할 만큼
#    큰 출력을 생성할 것으로 예상되지 않으므로, 검사할 필요가 없습니다.
TOOLS_EXCLUDED_FROM_EVICTION = (
    "ls",
    "glob",
    "grep",
    "read_file",
    "edit_file",
    "write_file",
)


TOO_LARGE_TOOL_MSG = """Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}

You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.

You can do this by specifying an offset and limit in the read_file tool call. For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

Here is a preview showing the head and tail of the result (lines of the form `... [N lines truncated] ...` indicate omitted lines in the middle of the content):

{content_sample}
"""


def _create_content_preview(content_str: str, *, head_lines: int = 5, tail_lines: int = 5) -> str:
    """콘텐츠의 머리와 꼬리 부분을 잘림 마커와 함께 보여주는 미리보기를 생성합니다.

    대용량 도구 결과가 파일시스템으로 제거될 때 사용자에게 내용의
    개요를 제공하기 위해 사용됩니다. 파일의 시작과 끝 부분만 표시하고
    중간은 생략 표시로 대체합니다.

    Args:
        content_str: 미리보기할 전체 콘텐츠 문자열.
        head_lines: 시작부터 표시할 줄 수.
        tail_lines: 끝에서 표시할 줄 수.

    Returns:
        줄 번호가 포함된 포맷된 미리보기 문자열.
    """
    lines = content_str.splitlines()

    # 파일이 충분히 작으면 전체 표시
    if len(lines) <= head_lines + tail_lines:
        # 각 줄을 1000자로 제한
        preview_lines = [line[:1000] for line in lines]
        return format_content_with_line_numbers(preview_lines, start_line=1)

    # 머리와 꼬리 부분을 잘림 마커와 함께 표시
    head = [line[:1000] for line in lines[:head_lines]]
    tail = [line[:1000] for line in lines[-tail_lines:]]

    head_sample = format_content_with_line_numbers(head, start_line=1)
    truncation_notice = f"\n... [{len(lines) - head_lines - tail_lines} lines truncated] ...\n"
    tail_sample = format_content_with_line_numbers(tail, start_line=len(lines) - tail_lines + 1)

    return head_sample + truncation_notice + tail_sample


class FilesystemMiddleware(AgentMiddleware):
    """에이전트에 파일시스템 및 선택적 실행 도구를 제공하는 미들웨어.

    이 미들웨어는 에이전트에 `ls`, `read_file`, `write_file`,
    `edit_file`, `glob`, `grep` 도구를 추가합니다.

    파일은 `BackendProtocol`을 구현하는 모든 백엔드에 저장할 수 있습니다.

    백엔드가 `SandboxBackendProtocol`을 구현하면 셸 명령 실행을 위한
    `execute` 도구도 추가됩니다.

    또한 큰 도구 결과가 토큰 임계값을 초과하면 파일시스템으로
    자동 제거(evict)하여 컨텍스트 윈도우 포화를 방지합니다.

    Args:
        backend: 파일 저장 및 선택적 실행을 위한 백엔드.

            제공되지 않으면 기본값은 `StateBackend`(에이전트 상태에 임시 저장).

            영구 저장 또는 하이브리드 설정의 경우 사용자 정의 라우트와
            함께 `CompositeBackend`를 사용하세요.

            실행 지원을 위해서는 `SandboxBackendProtocol`을 구현하는
            백엔드를 사용하세요.
        system_prompt: 사용자 정의 시스템 프롬프트 오버라이드 (선택사항).
        custom_tool_descriptions: 사용자 정의 도구 설명 오버라이드 (선택사항).
        tool_token_limit_before_evict: 도구 결과를 파일시스템으로 제거하기
            전의 토큰 제한.

            초과하면 구성된 백엔드를 사용하여 결과를 쓰고 잘린 미리보기와
            파일 참조로 대체합니다.

    사용 예시:
        ```python
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from deepagents.backends import StateBackend, StoreBackend, CompositeBackend
        from langchain.agents import create_agent

        # 임시 저장만 사용 (기본값, 실행 없음)
        agent = create_agent(middleware=[FilesystemMiddleware()])

        # 하이브리드 저장 사용 (임시 + /memories/에 영구 저장)
        backend = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend()})
        agent = create_agent(middleware=[FilesystemMiddleware(backend=backend)])

        # 샌드박스 백엔드 사용 (실행 지원)
        from my_sandbox import DockerSandboxBackend

        sandbox = DockerSandboxBackend(container_id="my-container")
        agent = create_agent(middleware=[FilesystemMiddleware(backend=sandbox)])
        ```
    """

    # 이 미들웨어의 상태 스키마
    state_schema = FilesystemState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES | None = None,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
        tool_token_limit_before_evict: int | None = 20000,
    ) -> None:
        """파일시스템 미들웨어를 초기화합니다.

        Args:
            backend: 파일 저장 및 선택적 실행을 위한 백엔드 또는 팩토리 콜러블.
                제공되지 않으면 StateBackend가 기본값.
            system_prompt: 사용자 정의 시스템 프롬프트 오버라이드 (선택사항).
            custom_tool_descriptions: 사용자 정의 도구 설명 오버라이드 (선택사항).
            tool_token_limit_before_evict: 도구 결과를 파일시스템으로 제거하기
                전의 토큰 제한 (선택사항).
        """
        # 제공된 백엔드 사용 또는 StateBackend 팩토리를 기본값으로 사용
        self.backend = backend if backend is not None else (lambda rt: StateBackend(rt))

        # 설정 저장 (private - 내부 구현 세부사항)
        self._custom_system_prompt = system_prompt
        self._custom_tool_descriptions = custom_tool_descriptions or {}
        self._tool_token_limit_before_evict = tool_token_limit_before_evict

        # 도구 목록 생성
        self.tools = [
            self._create_ls_tool(),          # 디렉토리 목록 조회
            self._create_read_file_tool(),   # 파일 읽기
            self._create_write_file_tool(),  # 파일 쓰기
            self._create_edit_file_tool(),   # 파일 편집
            self._create_glob_tool(),        # 패턴 매칭 검색
            self._create_grep_tool(),        # 텍스트 검색
            self._create_execute_tool(),     # 명령 실행 (샌드박스 백엔드 필요)
        ]

    def _get_backend(self, runtime: ToolRuntime) -> BackendProtocol:
        """백엔드 또는 팩토리에서 해결된 백엔드 인스턴스를 가져옵니다.

        Args:
            runtime: 도구 런타임 컨텍스트.

        Returns:
            해결된 백엔드 인스턴스.
        """
        # 백엔드가 콜러블(팩토리)이면 호출하여 인스턴스 생성
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    def _create_ls_tool(self) -> BaseTool:
        """ls (파일 목록 조회) 도구를 생성합니다."""
        tool_description = self._custom_tool_descriptions.get("ls") or LIST_FILES_TOOL_DESCRIPTION

        def sync_ls(
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Absolute path to the directory to list. Must be absolute, not relative."],
        ) -> str:
            """ls 도구의 동기 래퍼."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = _validate_path(path)
            except ValueError as e:
                return f"Error: {e}"
            # 백엔드에서 디렉토리 정보 조회
            infos = resolved_backend.ls_info(validated_path)
            # 경로만 추출
            paths = [fi.get("path", "") for fi in infos]
            # 결과가 너무 길면 잘라냄
            result = truncate_if_too_long(paths)
            return str(result)

        async def async_ls(
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Absolute path to the directory to list. Must be absolute, not relative."],
        ) -> str:
            """ls 도구의 비동기 래퍼."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = _validate_path(path)
            except ValueError as e:
                return f"Error: {e}"
            # 백엔드에서 디렉토리 정보 비동기 조회
            infos = await resolved_backend.als_info(validated_path)
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        return StructuredTool.from_function(
            name="ls",
            description=tool_description,
            func=sync_ls,
            coroutine=async_ls,
        )

    def _create_read_file_tool(self) -> BaseTool:
        """read_file (파일 읽기) 도구를 생성합니다."""
        tool_description = self._custom_tool_descriptions.get("read_file") or READ_FILE_TOOL_DESCRIPTION
        token_limit = self._tool_token_limit_before_evict

        def sync_read_file(
            file_path: Annotated[str, "Absolute path to the file to read. Must be absolute, not relative."],
            runtime: ToolRuntime[None, FilesystemState],
            offset: Annotated[int, "Line number to start reading from (0-indexed). Use for pagination of large files."] = DEFAULT_READ_OFFSET,
            limit: Annotated[int, "Maximum number of lines to read. Use for pagination of large files."] = DEFAULT_READ_LIMIT,
        ) -> str:
            """read_file 도구의 동기 래퍼."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = _validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"

            # 백엔드에서 파일 읽기
            result = resolved_backend.read(validated_path, offset=offset, limit=limit)

            # 줄 수 제한 적용
            lines = result.splitlines(keepends=True)
            if len(lines) > limit:
                lines = lines[:limit]
                result = "".join(lines)

            # 결과가 토큰 임계값을 초과하면 필요시 잘라냄
            if token_limit and len(result) >= NUM_CHARS_PER_TOKEN * token_limit:
                # 최종 결과가 임계값 이하가 되도록 잘림 메시지 길이 계산
                truncation_msg = READ_FILE_TRUNCATION_MSG.format(file_path=validated_path)
                max_content_length = NUM_CHARS_PER_TOKEN * token_limit - len(truncation_msg)
                result = result[:max_content_length]
                result += truncation_msg

            return result

        async def async_read_file(
            file_path: Annotated[str, "Absolute path to the file to read. Must be absolute, not relative."],
            runtime: ToolRuntime[None, FilesystemState],
            offset: Annotated[int, "Line number to start reading from (0-indexed). Use for pagination of large files."] = DEFAULT_READ_OFFSET,
            limit: Annotated[int, "Maximum number of lines to read. Use for pagination of large files."] = DEFAULT_READ_LIMIT,
        ) -> str:
            """read_file 도구의 비동기 래퍼."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = _validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"

            # 백엔드에서 파일 비동기 읽기
            result = await resolved_backend.aread(validated_path, offset=offset, limit=limit)

            # 줄 수 제한 적용
            lines = result.splitlines(keepends=True)
            if len(lines) > limit:
                lines = lines[:limit]
                result = "".join(lines)

            # 결과가 토큰 임계값을 초과하면 필요시 잘라냄
            if token_limit and len(result) >= NUM_CHARS_PER_TOKEN * token_limit:
                truncation_msg = READ_FILE_TRUNCATION_MSG.format(file_path=validated_path)
                max_content_length = NUM_CHARS_PER_TOKEN * token_limit - len(truncation_msg)
                result = result[:max_content_length]
                result += truncation_msg

            return result

        return StructuredTool.from_function(
            name="read_file",
            description=tool_description,
            func=sync_read_file,
            coroutine=async_read_file,
        )

    def _create_write_file_tool(self) -> BaseTool:
        """write_file (파일 쓰기) 도구를 생성합니다."""
        tool_description = self._custom_tool_descriptions.get("write_file") or WRITE_FILE_TOOL_DESCRIPTION

        def sync_write_file(
            file_path: Annotated[str, "Absolute path where the file should be created. Must be absolute, not relative."],
            content: Annotated[str, "The text content to write to the file. This parameter is required."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> Command | str:
            """write_file 도구의 동기 래퍼."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = _validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"

            # 백엔드에 파일 쓰기
            res: WriteResult = resolved_backend.write(validated_path, content)
            if res.error:
                return res.error

            # 백엔드가 상태 업데이트를 반환하면 Command로 래핑하여 ToolMessage 포함
            if res.files_update is not None:
                return Command(
                    update={
                        "files": res.files_update,
                        "messages": [
                            ToolMessage(
                                content=f"Updated file {res.path}",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                    }
                )
            return f"Updated file {res.path}"

        async def async_write_file(
            file_path: Annotated[str, "Absolute path where the file should be created. Must be absolute, not relative."],
            content: Annotated[str, "The text content to write to the file. This parameter is required."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> Command | str:
            """write_file 도구의 비동기 래퍼."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = _validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"

            # 백엔드에 파일 비동기 쓰기
            res: WriteResult = await resolved_backend.awrite(validated_path, content)
            if res.error:
                return res.error

            # 백엔드가 상태 업데이트를 반환하면 Command로 래핑하여 ToolMessage 포함
            if res.files_update is not None:
                return Command(
                    update={
                        "files": res.files_update,
                        "messages": [
                            ToolMessage(
                                content=f"Updated file {res.path}",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                    }
                )
            return f"Updated file {res.path}"

        return StructuredTool.from_function(
            name="write_file",
            description=tool_description,
            func=sync_write_file,
            coroutine=async_write_file,
        )

    def _create_edit_file_tool(self) -> BaseTool:
        """edit_file (파일 편집) 도구를 생성합니다."""
        tool_description = self._custom_tool_descriptions.get("edit_file") or EDIT_FILE_TOOL_DESCRIPTION

        def sync_edit_file(
            file_path: Annotated[str, "Absolute path to the file to edit. Must be absolute, not relative."],
            old_string: Annotated[str, "The exact text to find and replace. Must be unique in the file unless replace_all is True."],
            new_string: Annotated[str, "The text to replace old_string with. Must be different from old_string."],
            runtime: ToolRuntime[None, FilesystemState],
            *,
            replace_all: Annotated[bool, "If True, replace all occurrences of old_string. If False (default), old_string must be unique."] = False,
        ) -> Command | str:
            """edit_file 도구의 동기 래퍼."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = _validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"

            # 백엔드에서 파일 편집 수행
            res: EditResult = resolved_backend.edit(validated_path, old_string, new_string, replace_all=replace_all)
            if res.error:
                return res.error

            # 상태 업데이트가 있으면 Command로 래핑
            if res.files_update is not None:
                return Command(
                    update={
                        "files": res.files_update,
                        "messages": [
                            ToolMessage(
                                content=f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                    }
                )
            return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

        async def async_edit_file(
            file_path: Annotated[str, "Absolute path to the file to edit. Must be absolute, not relative."],
            old_string: Annotated[str, "The exact text to find and replace. Must be unique in the file unless replace_all is True."],
            new_string: Annotated[str, "The text to replace old_string with. Must be different from old_string."],
            runtime: ToolRuntime[None, FilesystemState],
            *,
            replace_all: Annotated[bool, "If True, replace all occurrences of old_string. If False (default), old_string must be unique."] = False,
        ) -> Command | str:
            """edit_file 도구의 비동기 래퍼."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = _validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"

            # 백엔드에서 파일 비동기 편집 수행
            res: EditResult = await resolved_backend.aedit(validated_path, old_string, new_string, replace_all=replace_all)
            if res.error:
                return res.error

            # 상태 업데이트가 있으면 Command로 래핑
            if res.files_update is not None:
                return Command(
                    update={
                        "files": res.files_update,
                        "messages": [
                            ToolMessage(
                                content=f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ],
                    }
                )
            return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

        return StructuredTool.from_function(
            name="edit_file",
            description=tool_description,
            func=sync_edit_file,
            coroutine=async_edit_file,
        )

    def _create_glob_tool(self) -> BaseTool:
        """glob (패턴 매칭 파일 검색) 도구를 생성합니다."""
        tool_description = self._custom_tool_descriptions.get("glob") or GLOB_TOOL_DESCRIPTION

        def sync_glob(
            pattern: Annotated[str, "Glob pattern to match files (e.g., '**/*.py', '*.txt', '/subdir/**/*.md')."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Base directory to search from. Defaults to root '/'."] = "/",
        ) -> str:
            """glob 도구의 동기 래퍼."""
            resolved_backend = self._get_backend(runtime)
            # 패턴에 매칭되는 파일 정보 조회
            infos = resolved_backend.glob_info(pattern, path=path)
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        async def async_glob(
            pattern: Annotated[str, "Glob pattern to match files (e.g., '**/*.py', '*.txt', '/subdir/**/*.md')."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Base directory to search from. Defaults to root '/'."] = "/",
        ) -> str:
            """glob 도구의 비동기 래퍼."""
            resolved_backend = self._get_backend(runtime)
            # 패턴에 매칭되는 파일 정보 비동기 조회
            infos = await resolved_backend.aglob_info(pattern, path=path)
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        return StructuredTool.from_function(
            name="glob",
            description=tool_description,
            func=sync_glob,
            coroutine=async_glob,
        )

    def _create_grep_tool(self) -> BaseTool:
        """grep (텍스트 패턴 검색) 도구를 생성합니다."""
        tool_description = self._custom_tool_descriptions.get("grep") or GREP_TOOL_DESCRIPTION

        def sync_grep(
            pattern: Annotated[str, "Text pattern to search for (literal string, not regex)."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str | None, "Directory to search in. Defaults to current working directory."] = None,
            glob: Annotated[str | None, "Glob pattern to filter which files to search (e.g., '*.py')."] = None,
            output_mode: Annotated[
                Literal["files_with_matches", "content", "count"],
                "Output format: 'files_with_matches' (file paths only, default), 'content' (matching lines with context), 'count' (match counts per file).",
            ] = "files_with_matches",
        ) -> str:
            """grep 도구의 동기 래퍼."""
            resolved_backend = self._get_backend(runtime)
            # 백엔드에서 패턴 검색 수행
            raw = resolved_backend.grep_raw(pattern, path=path, glob=glob)
            if isinstance(raw, str):
                return raw
            # 출력 모드에 따라 결과 포맷팅
            formatted = format_grep_matches(raw, output_mode)
            return truncate_if_too_long(formatted)  # type: ignore[arg-type]

        async def async_grep(
            pattern: Annotated[str, "Text pattern to search for (literal string, not regex)."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str | None, "Directory to search in. Defaults to current working directory."] = None,
            glob: Annotated[str | None, "Glob pattern to filter which files to search (e.g., '*.py')."] = None,
            output_mode: Annotated[
                Literal["files_with_matches", "content", "count"],
                "Output format: 'files_with_matches' (file paths only, default), 'content' (matching lines with context), 'count' (match counts per file).",
            ] = "files_with_matches",
        ) -> str:
            """grep 도구의 비동기 래퍼."""
            resolved_backend = self._get_backend(runtime)
            # 백엔드에서 패턴 비동기 검색 수행
            raw = await resolved_backend.agrep_raw(pattern, path=path, glob=glob)
            if isinstance(raw, str):
                return raw
            # 출력 모드에 따라 결과 포맷팅
            formatted = format_grep_matches(raw, output_mode)
            return truncate_if_too_long(formatted)  # type: ignore[arg-type]

        return StructuredTool.from_function(
            name="grep",
            description=tool_description,
            func=sync_grep,
            coroutine=async_grep,
        )

    def _create_execute_tool(self) -> BaseTool:
        """샌드박스 명령 실행을 위한 execute 도구를 생성합니다."""
        tool_description = self._custom_tool_descriptions.get("execute") or EXECUTE_TOOL_DESCRIPTION

        def sync_execute(
            command: Annotated[str, "Shell command to execute in the sandbox environment."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> str:
            """Synchronous wrapper for execute tool."""
            resolved_backend = self._get_backend(runtime)

            # Runtime check - fail gracefully if not supported
            if not _supports_execution(resolved_backend):
                return (
                    "Error: Execution not available. This agent's backend "
                    "does not support command execution (SandboxBackendProtocol). "
                    "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
                )

            try:
                result = resolved_backend.execute(command)
            except NotImplementedError as e:
                # Handle case where execute() exists but raises NotImplementedError
                return f"Error: Execution not available. {e}"

            # Format output for LLM consumption
            parts = [result.output]

            if result.exit_code is not None:
                status = "succeeded" if result.exit_code == 0 else "failed"
                parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

            if result.truncated:
                parts.append("\n[Output was truncated due to size limits]")

            return "".join(parts)

        async def async_execute(
            command: Annotated[str, "Shell command to execute in the sandbox environment."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> str:
            """Asynchronous wrapper for execute tool."""
            resolved_backend = self._get_backend(runtime)

            # Runtime check - fail gracefully if not supported
            if not _supports_execution(resolved_backend):
                return (
                    "Error: Execution not available. This agent's backend "
                    "does not support command execution (SandboxBackendProtocol). "
                    "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
                )

            try:
                result = await resolved_backend.aexecute(command)
            except NotImplementedError as e:
                # Handle case where execute() exists but raises NotImplementedError
                return f"Error: Execution not available. {e}"

            # Format output for LLM consumption
            parts = [result.output]

            if result.exit_code is not None:
                status = "succeeded" if result.exit_code == 0 else "failed"
                parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

            if result.truncated:
                parts.append("\n[Output was truncated due to size limits]")

            return "".join(parts)

        return StructuredTool.from_function(
            name="execute",
            description=tool_description,
            func=sync_execute,
            coroutine=async_execute,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Update the system prompt and filter tools based on backend capabilities.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)
            backend_supports_execution = _supports_execution(backend)

            # If execute tool exists but backend doesn't support it, filter it out
            if not backend_supports_execution:
                filtered_tools = [tool for tool in request.tools if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"]
                request = request.override(tools=filtered_tools)
                has_execute_tool = False

        # Use custom system prompt if provided, otherwise generate dynamically
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        else:
            # Build dynamic system prompt based on available tools
            prompt_parts = [FILESYSTEM_SYSTEM_PROMPT]

            # Add execution instructions if execute tool is available
            if has_execute_tool and backend_supports_execution:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)

            system_prompt = "\n\n".join(prompt_parts)

        if system_prompt:
            new_system_message = append_to_system_message(request.system_message, system_prompt)
            request = request.override(system_message=new_system_message)

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Update the system prompt and filter tools based on backend capabilities.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)
            backend_supports_execution = _supports_execution(backend)

            # If execute tool exists but backend doesn't support it, filter it out
            if not backend_supports_execution:
                filtered_tools = [tool for tool in request.tools if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"]
                request = request.override(tools=filtered_tools)
                has_execute_tool = False

        # Use custom system prompt if provided, otherwise generate dynamically
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        else:
            # Build dynamic system prompt based on available tools
            prompt_parts = [FILESYSTEM_SYSTEM_PROMPT]

            # Add execution instructions if execute tool is available
            if has_execute_tool and backend_supports_execution:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)

            system_prompt = "\n\n".join(prompt_parts)

        if system_prompt:
            new_system_message = append_to_system_message(request.system_message, system_prompt)
            request = request.override(system_message=new_system_message)

        return await handler(request)

    def _process_large_message(
        self,
        message: ToolMessage,
        resolved_backend: BackendProtocol,
    ) -> tuple[ToolMessage, dict[str, FileData] | None]:
        """Process a large ToolMessage by evicting its content to filesystem.

        Args:
            message: The ToolMessage with large content to evict.
            resolved_backend: The filesystem backend to write the content to.

        Returns:
            A tuple of (processed_message, files_update):
            - processed_message: New ToolMessage with truncated content and file reference
            - files_update: Dict of file updates to apply to state, or None if eviction failed

        Note:
            The entire content is converted to string, written to /large_tool_results/{tool_call_id},
            and replaced with a truncated preview plus file reference. The replacement is always
            returned as a plain string for consistency, regardless of original content type.

            ToolMessage supports multimodal content blocks (images, audio, etc.), but these are
            uncommon in tool results. For simplicity, all content is stringified and evicted.
            The model can recover by reading the offloaded file from the backend.
        """
        # Early exit if eviction not configured
        if not self._tool_token_limit_before_evict:
            return message, None

        # Convert content to string once for both size check and eviction
        # Special case: single text block - extract text directly for readability
        if (
            isinstance(message.content, list)
            and len(message.content) == 1
            and isinstance(message.content[0], dict)
            and message.content[0].get("type") == "text"
            and "text" in message.content[0]
        ):
            content_str = str(message.content[0]["text"])
        elif isinstance(message.content, str):
            content_str = message.content
        else:
            # Multiple blocks or non-text content - stringify entire structure
            content_str = str(message.content)

        # Check if content exceeds eviction threshold
        if len(content_str) <= NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict:
            return message, None

        # Write content to filesystem
        sanitized_id = sanitize_tool_call_id(message.tool_call_id)
        file_path = f"/large_tool_results/{sanitized_id}"
        result = resolved_backend.write(file_path, content_str)
        if result.error:
            return message, None

        # Create preview showing head and tail of the result
        content_sample = _create_content_preview(content_str)
        replacement_text = TOO_LARGE_TOOL_MSG.format(
            tool_call_id=message.tool_call_id,
            file_path=file_path,
            content_sample=content_sample,
        )

        # Always return as plain string after eviction
        processed_message = ToolMessage(
            content=replacement_text,
            tool_call_id=message.tool_call_id,
            name=message.name,
        )
        return processed_message, result.files_update

    async def _aprocess_large_message(
        self,
        message: ToolMessage,
        resolved_backend: BackendProtocol,
    ) -> tuple[ToolMessage, dict[str, FileData] | None]:
        """Async version of _process_large_message.

        Uses async backend methods to avoid sync calls in async context.
        See _process_large_message for full documentation.
        """
        # Early exit if eviction not configured
        if not self._tool_token_limit_before_evict:
            return message, None

        # Convert content to string once for both size check and eviction
        # Special case: single text block - extract text directly for readability
        if (
            isinstance(message.content, list)
            and len(message.content) == 1
            and isinstance(message.content[0], dict)
            and message.content[0].get("type") == "text"
            and "text" in message.content[0]
        ):
            content_str = str(message.content[0]["text"])
        elif isinstance(message.content, str):
            content_str = message.content
        else:
            # Multiple blocks or non-text content - stringify entire structure
            content_str = str(message.content)

        if len(content_str) <= NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict:
            return message, None

        # Write content to filesystem using async method
        sanitized_id = sanitize_tool_call_id(message.tool_call_id)
        file_path = f"/large_tool_results/{sanitized_id}"
        result = await resolved_backend.awrite(file_path, content_str)
        if result.error:
            return message, None

        # Create preview showing head and tail of the result
        content_sample = _create_content_preview(content_str)
        replacement_text = TOO_LARGE_TOOL_MSG.format(
            tool_call_id=message.tool_call_id,
            file_path=file_path,
            content_sample=content_sample,
        )

        # Always return as plain string after eviction
        processed_message = ToolMessage(
            content=replacement_text,
            tool_call_id=message.tool_call_id,
            name=message.name,
        )
        return processed_message, result.files_update

    def _intercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime: ToolRuntime) -> ToolMessage | Command:
        """Intercept and process large tool results before they're added to state.

        Args:
            tool_result: The tool result to potentially evict (ToolMessage or Command).
            runtime: The tool runtime providing access to the filesystem backend.

        Returns:
            Either the original result (if small enough) or a Command with evicted
            content written to filesystem and truncated message.

        Note:
            Handles both single ToolMessage results and Command objects containing
            multiple messages. Large content is automatically offloaded to filesystem
            to prevent context window overflow.
        """
        if isinstance(tool_result, ToolMessage):
            resolved_backend = self._get_backend(runtime)
            processed_message, files_update = self._process_large_message(
                tool_result,
                resolved_backend,
            )
            return (
                Command(
                    update={
                        "files": files_update,
                        "messages": [processed_message],
                    }
                )
                if files_update is not None
                else processed_message
            )

        if isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            command_messages = update.get("messages", [])
            accumulated_file_updates = dict(update.get("files", {}))
            resolved_backend = self._get_backend(runtime)
            processed_messages = []
            for message in command_messages:
                if not isinstance(message, ToolMessage):
                    processed_messages.append(message)
                    continue

                processed_message, files_update = self._process_large_message(
                    message,
                    resolved_backend,
                )
                processed_messages.append(processed_message)
                if files_update is not None:
                    accumulated_file_updates.update(files_update)
            return Command(update={**update, "messages": processed_messages, "files": accumulated_file_updates})
        raise AssertionError(f"Unreachable code reached in _intercept_large_tool_result: for tool_result of type {type(tool_result)}")

    async def _aintercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime: ToolRuntime) -> ToolMessage | Command:
        """Async version of _intercept_large_tool_result.

        Uses async backend methods to avoid sync calls in async context.
        See _intercept_large_tool_result for full documentation.
        """
        if isinstance(tool_result, ToolMessage):
            resolved_backend = self._get_backend(runtime)
            processed_message, files_update = await self._aprocess_large_message(
                tool_result,
                resolved_backend,
            )
            return (
                Command(
                    update={
                        "files": files_update,
                        "messages": [processed_message],
                    }
                )
                if files_update is not None
                else processed_message
            )

        if isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            command_messages = update.get("messages", [])
            accumulated_file_updates = dict(update.get("files", {}))
            resolved_backend = self._get_backend(runtime)
            processed_messages = []
            for message in command_messages:
                if not isinstance(message, ToolMessage):
                    processed_messages.append(message)
                    continue

                processed_message, files_update = await self._aprocess_large_message(
                    message,
                    resolved_backend,
                )
                processed_messages.append(processed_message)
                if files_update is not None:
                    accumulated_file_updates.update(files_update)
            return Command(update={**update, "messages": processed_messages, "files": accumulated_file_updates})
        raise AssertionError(f"Unreachable code reached in _aintercept_large_tool_result: for tool_result of type {type(tool_result)}")

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Check the size of the tool call result and evict to filesystem if too large.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The raw ToolMessage, or a pseudo tool message with the ToolResult in state.
        """
        if self._tool_token_limit_before_evict is None or request.tool_call["name"] in TOOLS_EXCLUDED_FROM_EVICTION:
            return handler(request)

        tool_result = handler(request)
        return self._intercept_large_tool_result(tool_result, request.runtime)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """(async)Check the size of the tool call result and evict to filesystem if too large.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The raw ToolMessage, or a pseudo tool message with the ToolResult in state.
        """
        if self._tool_token_limit_before_evict is None or request.tool_call["name"] in TOOLS_EXCLUDED_FROM_EVICTION:
            return await handler(request)

        tool_result = await handler(request)
        return await self._aintercept_large_tool_result(tool_result, request.runtime)
