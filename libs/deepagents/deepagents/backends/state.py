"""
모듈명: state.py
설명: LangGraph 에이전트 상태에 파일을 저장하는 백엔드 (임시 저장소)

이 모듈은 LangGraph의 상태 관리 및 체크포인팅 기능을 활용하여
에이전트 상태에 파일을 저장합니다. 파일은 대화 스레드 내에서만 유지되며
스레드 간에는 공유되지 않습니다.

주요 특징:
- LangGraph 상태 기반의 임시 파일 저장소
- 대화 스레드 범위 내 파일 지속성
- 에이전트 단계마다 자동 체크포인팅
- Command 객체를 통한 상태 업데이트 (uses_state=True 플래그)

주요 클래스:
- StateBackend: 상태 기반 파일 저장소 백엔드

사용 예시:
    ```python
    from deepagents.backends.state import StateBackend
    from langchain.tools import ToolRuntime

    backend = StateBackend(runtime)
    content = backend.read("/path/to/file.txt")
    result = backend.write("/path/to/new_file.txt", "content")
    ```

참고:
    LangGraph 상태는 직접 변경이 아닌 Command 객체를 통해 업데이트되어야 합니다.
    따라서 이 백엔드의 쓰기 작업은 None 대신 Command 객체를 반환합니다.
"""

from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)
from deepagents.backends.utils import (
    _glob_search_files,
    create_file_data,
    file_data_to_string,
    format_read_response,
    grep_matches_from_files,
    perform_string_replacement,
    update_file_data,
)

if TYPE_CHECKING:
    from langchain.tools import ToolRuntime


class StateBackend(BackendProtocol):
    """Backend that stores files in agent state (ephemeral).

    Uses LangGraph's state management and checkpointing. Files persist within
    a conversation thread but not across threads. State is automatically
    checkpointed after each agent step.

    Special handling: Since LangGraph state must be updated via Command objects
    (not direct mutation), operations return Command objects instead of None.
    This is indicated by the uses_state=True flag.
    """

    def __init__(self, runtime: "ToolRuntime"):
        """Initialize StateBackend with runtime."""
        self.runtime = runtime

    def ls_info(self, path: str) -> list[FileInfo]:
        """List files and directories in the specified directory (non-recursive).

        Args:
            path: Absolute path to directory.

        Returns:
            List of FileInfo-like dicts for files and directories directly in the directory.
            Directories have a trailing / in their path and is_dir=True.
        """
        files = self.runtime.state.get("files", {})
        infos: list[FileInfo] = []
        subdirs: set[str] = set()

        # Normalize path to have trailing slash for proper prefix matching
        normalized_path = path if path.endswith("/") else path + "/"

        for k, fd in files.items():
            # Check if file is in the specified directory or a subdirectory
            if not k.startswith(normalized_path):
                continue

            # Get the relative path after the directory
            relative = k[len(normalized_path) :]

            # If relative path contains '/', it's in a subdirectory
            if "/" in relative:
                # Extract the immediate subdirectory name
                subdir_name = relative.split("/")[0]
                subdirs.add(normalized_path + subdir_name + "/")
                continue

            # This is a file directly in the current directory
            size = len("\n".join(fd.get("content", [])))
            infos.append(
                {
                    "path": k,
                    "is_dir": False,
                    "size": int(size),
                    "modified_at": fd.get("modified_at", ""),
                }
            )

        # Add directories to the results
        for subdir in sorted(subdirs):
            infos.append(
                {
                    "path": subdir,
                    "is_dir": True,
                    "size": 0,
                    "modified_at": "",
                }
            )

        infos.sort(key=lambda x: x.get("path", ""))
        return infos

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.

        Args:
            file_path: Absolute file path.
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            Formatted file content with line numbers, or error message.
        """
        files = self.runtime.state.get("files", {})
        file_data = files.get(file_path)

        if file_data is None:
            return f"Error: File '{file_path}' not found"

        return format_read_response(file_data, offset, limit)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file with content.
        Returns WriteResult with files_update to update LangGraph state.
        """
        files = self.runtime.state.get("files", {})

        if file_path in files:
            return WriteResult(error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path.")

        new_file_data = create_file_data(content)
        return WriteResult(path=file_path, files_update={file_path: new_file_data})

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences.
        Returns EditResult with files_update and occurrences.
        """
        files = self.runtime.state.get("files", {})
        file_data = files.get(file_path)

        if file_data is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)

        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)
        return EditResult(path=file_path, files_update={file_path: new_file_data}, occurrences=int(occurrences))

    def grep_raw(
        self,
        pattern: str,
        path: str = "/",
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        files = self.runtime.state.get("files", {})
        return grep_matches_from_files(files, pattern, path, glob)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Get FileInfo for files matching glob pattern."""
        files = self.runtime.state.get("files", {})
        result = _glob_search_files(files, pattern, path)
        if result == "No files found":
            return []
        paths = result.split("\n")
        infos: list[FileInfo] = []
        for p in paths:
            fd = files.get(p)
            size = len("\n".join(fd.get("content", []))) if fd else 0
            infos.append(
                {
                    "path": p,
                    "is_dir": False,
                    "size": int(size),
                    "modified_at": fd.get("modified_at", "") if fd else "",
                }
            )
        return infos

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to state.

        Args:
            files: List of (path, content) tuples to upload

        Returns:
            List of FileUploadResponse objects, one per input file
        """
        raise NotImplementedError(
            "StateBackend does not support upload_files yet. You can upload files "
            "directly by passing them in invoke if you're storing files in the memory."
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from state.

        Args:
            paths: List of file paths to download

        Returns:
            List of FileDownloadResponse objects, one per input path
        """
        state_files = self.runtime.state.get("files", {})
        responses: list[FileDownloadResponse] = []

        for path in paths:
            file_data = state_files.get(path)

            if file_data is None:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
                continue

            # Convert file data to bytes
            content_str = file_data_to_string(file_data)
            content_bytes = content_str.encode("utf-8")

            responses.append(FileDownloadResponse(path=path, content=content_bytes, error=None))

        return responses
