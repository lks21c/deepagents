"""
모듈명: composite.py
설명: 경로 접두사 기반으로 파일 작업을 라우팅하는 복합 백엔드

경로 접두사에 따라 서로 다른 백엔드로 작업을 라우팅합니다.
다른 경로에 대해 다른 저장소 전략이 필요할 때 사용합니다
(예: 임시 파일은 상태에, 메모리는 영구 저장소에).

주요 클래스:
    - CompositeBackend: 경로 접두사로 여러 백엔드에 파일 작업을 라우팅

사용 예시:
    ```python
    from deepagents.backends.composite import CompositeBackend
    from deepagents.backends.state import StateBackend
    from deepagents.backends.store import StoreBackend

    runtime = make_runtime()
    composite = CompositeBackend(
        default=StateBackend(runtime),
        routes={"/memories/": StoreBackend(runtime)}
    )

    # 기본 백엔드(StateBackend)로 라우팅
    composite.write("/temp.txt", "ephemeral")

    # /memories/ 접두사로 StoreBackend로 라우팅
    composite.write("/memories/note.md", "persistent")
    ```
"""

from collections import defaultdict

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.state import StateBackend


class CompositeBackend(BackendProtocol):
    """
    경로 접두사에 따라 서로 다른 백엔드로 파일 작업을 라우팅하는 백엔드입니다.

    경로를 라우트 접두사와 매칭하고(가장 긴 것부터) 해당 백엔드에 위임합니다.
    매칭되지 않는 경로는 기본 백엔드를 사용합니다.

    라우팅 알고리즘:
        1. 등록된 라우트를 길이 기준 내림차순으로 정렬
        2. 각 경로에 대해 가장 긴 매칭 접두사를 가진 백엔드 선택
        3. 매칭되는 접두사가 없으면 기본 백엔드 사용

    Attributes:
        default: 어떤 라우트와도 매칭되지 않는 경로를 위한 백엔드.
        routes: 경로 접두사와 백엔드의 매핑 (예: {"/memories/": store_backend}).
        sorted_routes: 올바른 매칭을 위해 길이순으로 정렬된 라우트(가장 긴 것부터).

    Examples:
        ```python
        composite = CompositeBackend(
            default=StateBackend(runtime),
            routes={
                "/memories/": StoreBackend(runtime),
                "/cache/": StoreBackend(runtime)
            }
        )

        # 기본 백엔드로 라우팅 (매칭 접두사 없음)
        composite.write("/temp.txt", "data")

        # /memories/ 라우트의 StoreBackend로 라우팅
        composite.write("/memories/note.txt", "data")
        ```
    """

    def __init__(
        self,
        default: BackendProtocol | StateBackend,
        routes: dict[str, BackendProtocol],
    ) -> None:
        """
        복합 백엔드를 초기화합니다.

        Args:
            default: 어떤 라우트와도 매칭되지 않는 경로를 위한 백엔드.
            routes: 경로 접두사와 백엔드의 매핑. 접두사는 "/"로 시작해야 하고
                "/"로 끝나야 합니다 (예: "/memories/").
        """
        # 기본 백엔드 - 매칭되는 라우트가 없을 때 사용
        self.default = default

        # 가상 라우트 - 경로 접두사별 백엔드 매핑
        self.routes = routes

        # 올바른 접두사 매칭을 위해 라우트를 길이순으로 정렬 (가장 긴 것부터)
        # 예: "/memories/notes/"가 "/memories/"보다 먼저 매칭되도록
        self.sorted_routes = sorted(routes.items(), key=lambda x: len(x[0]), reverse=True)

    def _get_backend_and_key(self, key: str) -> tuple[BackendProtocol, str]:
        """
        경로에 대한 백엔드를 가져오고 라우트 접두사를 제거합니다.

        이 메서드는 내부적으로 모든 파일 작업에서 사용되며,
        주어진 경로를 적절한 백엔드로 라우팅합니다.

        Args:
            key: 라우팅할 파일 경로.

        Returns:
            (backend, stripped_path) 튜플. stripped_path는 라우트 접두사가
            제거되었지만 선행 슬래시는 유지됩니다.

        Examples:
            "/memories/notes.txt" → (store_backend, "/notes.txt")
            "/memories/" → (store_backend, "/")
            "/temp.txt" → (default_backend, "/temp.txt")
        """
        # 길이 순서대로 라우트 확인 (가장 긴 것부터)
        for prefix, backend in self.sorted_routes:
            if key.startswith(prefix):
                # 전체 접두사를 제거하고 선행 슬래시가 유지되도록 함
                # 예: "/memories/notes.txt" → "/notes.txt"; "/memories/" → "/"
                suffix = key[len(prefix) :]
                stripped_key = f"/{suffix}" if suffix else "/"
                return backend, stripped_key

        # 매칭되는 라우트가 없으면 기본 백엔드 반환
        return self.default, key

    def ls_info(self, path: str) -> list[FileInfo]:
        """
        디렉토리 내용을 나열합니다 (비재귀적).

        경로가 라우트와 매칭되면 해당 백엔드만 조회합니다.
        경로가 "/"이면 기본 백엔드와 가상 라우트 디렉토리를 통합합니다.
        그 외에는 기본 백엔드를 조회합니다.

        라우팅 로직:
            1. 특정 라우트와 매칭 → 해당 백엔드만 조회
            2. 루트 경로("/") → 기본 백엔드 + 모든 라우트 디렉토리 통합
            3. 기타 → 기본 백엔드만 조회

        Args:
            path: "/"로 시작하는 절대 디렉토리 경로.

        Returns:
            FileInfo 딕셔너리 목록. 디렉토리는 후행 "/"와 is_dir=True를 가집니다.
            반환된 경로에는 라우트 접두사가 복원됩니다.

        Examples:
            ```python
            # 루트에서 모든 백엔드의 내용과 가상 디렉토리 조회
            infos = composite.ls_info("/")

            # /memories/ 라우트의 백엔드에서만 조회
            infos = composite.ls_info("/memories/")
            ```
        """
        # 경로가 특정 라우트와 매칭되는지 확인
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                # 매칭된 라우트 백엔드만 조회
                suffix = path[len(route_prefix) :]
                search_path = f"/{suffix}" if suffix else "/"
                infos = backend.ls_info(search_path)
                # 결과에 라우트 접두사를 복원
                prefixed: list[FileInfo] = []
                for fi in infos:
                    fi = dict(fi)
                    fi["path"] = f"{route_prefix[:-1]}{fi['path']}"
                    prefixed.append(fi)
                return prefixed

        # 루트에서는 기본 백엔드와 모든 라우트 백엔드를 통합
        if path == "/":
            results: list[FileInfo] = []
            results.extend(self.default.ls_info(path))
            for route_prefix, backend in self.sorted_routes:
                # 라우트 자체를 디렉토리로 추가 (예: /memories/)
                results.append(
                    {
                        "path": route_prefix,
                        "is_dir": True,
                        "size": 0,
                        "modified_at": "",
                    }
                )

            # 결정적 순서를 위해 경로로 정렬
            results.sort(key=lambda x: x.get("path", ""))
            return results

        # 경로가 라우트와 매칭되지 않음: 기본 백엔드만 조회
        return self.default.ls_info(path)

    async def als_info(self, path: str) -> list[FileInfo]:
        """ls_info의 비동기 버전입니다."""
        # 경로가 특정 라우트와 매칭되는지 확인
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                # Query only the matching routed backend
                suffix = path[len(route_prefix) :]
                search_path = f"/{suffix}" if suffix else "/"
                infos = await backend.als_info(search_path)
                prefixed: list[FileInfo] = []
                for fi in infos:
                    fi = dict(fi)
                    fi["path"] = f"{route_prefix[:-1]}{fi['path']}"
                    prefixed.append(fi)
                return prefixed

        # At root, aggregate default and all routed backends
        if path == "/":
            results: list[FileInfo] = []
            results.extend(await self.default.als_info(path))
            for route_prefix, backend in self.sorted_routes:
                # Add the route itself as a directory (e.g., /memories/)
                results.append(
                    {
                        "path": route_prefix,
                        "is_dir": True,
                        "size": 0,
                        "modified_at": "",
                    }
                )

            results.sort(key=lambda x: x.get("path", ""))
            return results

        # Path doesn't match a route: query only default backend
        return await self.default.als_info(path)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """
        파일 내용을 읽습니다. 적절한 백엔드로 라우팅됩니다.

        Args:
            file_path: 절대 파일 경로.
            offset: 읽기 시작할 라인 오프셋 (0-indexed).
            limit: 읽을 최대 라인 수.

        Returns:
            라인 번호가 포함된 형식의 파일 내용, 또는 오류 메시지.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.read(stripped_key, offset=offset, limit=limit)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """read의 비동기 버전입니다."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        return await backend.aread(stripped_key, offset=offset, limit=limit)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """
        파일에서 리터럴 텍스트 패턴을 검색합니다.

        경로에 따라 백엔드로 라우팅합니다: 특정 라우트는 해당 백엔드만 검색,
        "/" 또는 None은 모든 백엔드 검색, 그 외에는 기본 백엔드 검색.

        라우팅 로직:
            1. 특정 라우트와 매칭되는 경로 → 해당 백엔드만 검색
            2. None 또는 "/" → 기본 백엔드 + 모든 라우트 백엔드 검색 후 병합
            3. 기타 → 기본 백엔드만 검색

        Args:
            pattern: 검색할 리터럴 텍스트 (정규식이 아님).
            path: 검색할 디렉토리. None은 모든 백엔드 검색.
            glob: 파일 필터링을 위한 글롭 패턴 (예: "*.py", "**/*.txt").
                내용이 아닌 파일명으로 필터링.

        Returns:
            경로(라우트 접두사 복원됨), 라인(1-indexed), 텍스트를 담은
            GrepMatch 딕셔너리 목록. 실패 시 오류 문자열 반환.

        Examples:
            ```python
            # /memories/ 라우트의 백엔드에서만 검색
            matches = composite.grep_raw("TODO", path="/memories/")

            # 모든 백엔드에서 검색
            matches = composite.grep_raw("error", path="/")

            # 모든 백엔드에서 Python 파일만 검색
            matches = composite.grep_raw("import", path="/", glob="*.py")
            ```
        """
        # 경로가 특정 라우트를 대상으로 하면 해당 백엔드만 검색
        for route_prefix, backend in self.sorted_routes:
            if path is not None and path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1 :]
                raw = backend.grep_raw(pattern, search_path if search_path else "/", glob)
                if isinstance(raw, str):
                    return raw
                # 결과에 라우트 접두사 복원
                return [{**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw]

        # path가 None 또는 "/"이면 기본 백엔드와 모든 라우트 백엔드를 검색하고 병합
        # 그 외에는 기본 백엔드만 검색
        if path is None or path == "/":
            all_matches: list[GrepMatch] = []
            raw_default = self.default.grep_raw(pattern, path, glob)  # type: ignore[attr-defined]
            if isinstance(raw_default, str):
                # 오류 발생 시
                return raw_default
            all_matches.extend(raw_default)

            # 모든 라우트 백엔드에서 검색
            for route_prefix, backend in self.routes.items():
                raw = backend.grep_raw(pattern, "/", glob)
                if isinstance(raw, str):
                    # 오류 발생 시
                    return raw
                # 결과에 라우트 접두사 복원하여 추가
                all_matches.extend({**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw)

            return all_matches
        # 경로가 지정되었지만 라우트와 매칭되지 않음 - 기본 백엔드만 검색
        return self.default.grep_raw(pattern, path, glob)  # type: ignore[attr-defined]

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """
        grep_raw의 비동기 버전입니다.

        라우팅 동작과 파라미터에 대한 자세한 문서는 grep_raw()를 참조하세요.
        """
        # If path targets a specific route, search only that backend
        for route_prefix, backend in self.sorted_routes:
            if path is not None and path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1 :]
                raw = await backend.agrep_raw(pattern, search_path if search_path else "/", glob)
                if isinstance(raw, str):
                    return raw
                return [{**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw]

        # If path is None or "/", search default and all routed backends and merge
        # Otherwise, search only the default backend
        if path is None or path == "/":
            all_matches: list[GrepMatch] = []
            raw_default = await self.default.agrep_raw(pattern, path, glob)  # type: ignore[attr-defined]
            if isinstance(raw_default, str):
                # This happens if error occurs
                return raw_default
            all_matches.extend(raw_default)

            for route_prefix, backend in self.routes.items():
                raw = await backend.agrep_raw(pattern, "/", glob)
                if isinstance(raw, str):
                    # This happens if error occurs
                    return raw
                all_matches.extend({**m, "path": f"{route_prefix[:-1]}{m['path']}"} for m in raw)

            return all_matches
        # Path specified but doesn't match a route - search only default
        return await self.default.agrep_raw(pattern, path, glob)  # type: ignore[attr-defined]

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        results: list[FileInfo] = []

        # Route based on path, not pattern
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1 :]
                infos = backend.glob_info(pattern, search_path if search_path else "/")
                return [{**fi, "path": f"{route_prefix[:-1]}{fi['path']}"} for fi in infos]

        # Path doesn't match any specific route - search default backend AND all routed backends
        results.extend(self.default.glob_info(pattern, path))

        for route_prefix, backend in self.routes.items():
            infos = backend.glob_info(pattern, "/")
            results.extend({**fi, "path": f"{route_prefix[:-1]}{fi['path']}"} for fi in infos)

        # Deterministic ordering
        results.sort(key=lambda x: x.get("path", ""))
        return results

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """glob_info의 비동기 버전입니다."""
        results: list[FileInfo] = []

        # Route based on path, not pattern
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1 :]
                infos = await backend.aglob_info(pattern, search_path if search_path else "/")
                return [{**fi, "path": f"{route_prefix[:-1]}{fi['path']}"} for fi in infos]

        # Path doesn't match any specific route - search default backend AND all routed backends
        results.extend(await self.default.aglob_info(pattern, path))

        for route_prefix, backend in self.routes.items():
            infos = await backend.aglob_info(pattern, "/")
            results.extend({**fi, "path": f"{route_prefix[:-1]}{fi['path']}"} for fi in infos)

        # Deterministic ordering
        results.sort(key=lambda x: x.get("path", ""))
        return results

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """
        새 파일을 생성합니다. 적절한 백엔드로 라우팅됩니다.

        Args:
            file_path: 절대 파일 경로.
            content: 문자열 형태의 파일 내용.

        Returns:
            WriteResult: 성공 시 경로 정보, 또는 파일이 이미 존재하면 오류.

        Note:
            상태 기반 백엔드에서 업데이트가 발생하면 기본 백엔드의 상태에도
            병합하여 파일 목록이 변경사항을 반영하도록 합니다.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = backend.write(stripped_key, content)
        # 상태 기반 업데이트이고 기본 백엔드에 상태가 있으면 병합하여 목록이 변경을 반영하도록 함
        if res.files_update:
            try:
                runtime = getattr(self.default, "runtime", None)
                if runtime is not None:
                    state = runtime.state
                    files = state.get("files", {})
                    files.update(res.files_update)
                    state["files"] = files
            except Exception:
                pass
        return res

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """write의 비동기 버전입니다."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = await backend.awrite(stripped_key, content)
        # 상태 기반 업데이트이고 기본 백엔드에 상태가 있으면 병합하여 목록이 변경을 반영하도록 함
        if res.files_update:
            try:
                runtime = getattr(self.default, "runtime", None)
                if runtime is not None:
                    state = runtime.state
                    files = state.get("files", {})
                    files.update(res.files_update)
                    state["files"] = files
            except Exception:
                pass
        return res

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """
        파일을 편집합니다. 적절한 백엔드로 라우팅됩니다.

        Args:
            file_path: 절대 파일 경로.
            old_string: 찾아서 교체할 문자열.
            new_string: 교체할 문자열.
            replace_all: True이면 모든 발생을 교체.

        Returns:
            EditResult: 성공 시 경로와 교체 횟수, 또는 실패 시 오류 메시지.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = backend.edit(stripped_key, old_string, new_string, replace_all=replace_all)
        if res.files_update:
            try:
                runtime = getattr(self.default, "runtime", None)
                if runtime is not None:
                    state = runtime.state
                    files = state.get("files", {})
                    files.update(res.files_update)
                    state["files"] = files
            except Exception:
                pass
        return res

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """edit의 비동기 버전입니다."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        res = await backend.aedit(stripped_key, old_string, new_string, replace_all=replace_all)
        if res.files_update:
            try:
                runtime = getattr(self.default, "runtime", None)
                if runtime is not None:
                    state = runtime.state
                    files = state.get("files", {})
                    files.update(res.files_update)
                    state["files"] = files
            except Exception:
                pass
        return res

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """
        기본 백엔드를 통해 셸 명령을 실행합니다.

        명령 실행은 기본 백엔드에 위임됩니다. 기본 백엔드가
        SandboxBackendProtocol을 구현해야 합니다.

        Args:
            command: 실행할 셸 명령.

        Returns:
            ExecuteResponse: 출력, 종료 코드, 잘림 플래그를 포함.

        Raises:
            NotImplementedError: 기본 백엔드가 SandboxBackendProtocol을 구현하지 않는 경우.

        Examples:
            ```python
            composite = CompositeBackend(
                default=FilesystemBackend(root_dir="/tmp"),
                routes={"/memories/": StoreBackend(runtime)}
            )

            result = composite.execute("ls -la")
            ```
        """
        if isinstance(self.default, SandboxBackendProtocol):
            return self.default.execute(command)

        # execute 도구의 런타임 검사가 올바르게 작동하면 여기에 도달하지 않아야 하지만,
        # 안전 폴백으로 포함합니다.
        raise NotImplementedError(
            "기본 백엔드가 명령 실행을 지원하지 않습니다 (SandboxBackendProtocol). "
            "실행을 활성화하려면 SandboxBackendProtocol을 구현하는 기본 백엔드를 제공하세요."
        )

    async def aexecute(
        self,
        command: str,
    ) -> ExecuteResponse:
        """execute의 비동기 버전입니다."""
        if isinstance(self.default, SandboxBackendProtocol):
            return await self.default.aexecute(command)

        # execute 도구의 런타임 검사가 올바르게 작동하면 여기에 도달하지 않아야 하지만,
        # 안전 폴백으로 포함합니다.
        raise NotImplementedError(
            "기본 백엔드가 명령 실행을 지원하지 않습니다 (SandboxBackendProtocol). "
            "실행을 활성화하려면 SandboxBackendProtocol을 구현하는 기본 백엔드를 제공하세요."
        )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """
        여러 파일을 업로드합니다. 효율성을 위해 백엔드별로 배칭합니다.

        파일을 대상 백엔드별로 그룹화하고, 각 백엔드의 upload_files를
        해당 백엔드의 모든 파일과 함께 한 번 호출한 후, 결과를 원래 순서로 병합합니다.

        배칭 전략:
            1. 파일을 대상 백엔드별로 그룹화 (원래 인덱스 추적)
            2. 각 백엔드에 대해 한 번에 모든 파일 업로드 호출
            3. 응답을 원래 인덱스에 맞게 배치하여 입력 순서 유지

        Args:
            files: 업로드할 (경로, 내용) 튜플 목록.

        Returns:
            입력 파일당 하나씩 FileUploadResponse 객체 목록.
            응답 순서는 입력 순서와 일치합니다.
        """
        # 결과 목록 사전 할당
        results: list[FileUploadResponse | None] = [None] * len(files)

        # 백엔드별로 파일 그룹화, 원래 인덱스 추적
        from collections import defaultdict

        backend_batches: dict[BackendProtocol, list[tuple[int, str, bytes]]] = defaultdict(list)

        for idx, (path, content) in enumerate(files):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path, content))

        # 각 백엔드의 배치 처리
        for backend, batch in backend_batches.items():
            # 백엔드 호출을 위한 데이터 추출
            indices, stripped_paths, contents = zip(*batch, strict=False)
            batch_files = list(zip(stripped_paths, contents, strict=False))

            # 백엔드를 모든 파일과 함께 한 번 호출
            batch_responses = backend.upload_files(batch_files)

            # 원래 인덱스와 원래 경로로 응답 배치
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileUploadResponse(
                    path=files[orig_idx][0],  # 원래 경로
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return results  # type: ignore[return-value]

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """upload_files의 비동기 버전입니다."""
        # 결과 목록 사전 할당
        results: list[FileUploadResponse | None] = [None] * len(files)

        # 백엔드별로 파일 그룹화, 원래 인덱스 추적
        backend_batches: dict[BackendProtocol, list[tuple[int, str, bytes]]] = defaultdict(list)

        for idx, (path, content) in enumerate(files):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path, content))

        # Process each backend's batch
        for backend, batch in backend_batches.items():
            # Extract data for backend call
            indices, stripped_paths, contents = zip(*batch, strict=False)
            batch_files = list(zip(stripped_paths, contents, strict=False))

            # Call backend once with all its files
            batch_responses = await backend.aupload_files(batch_files)

            # Place responses at original indices with original paths
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileUploadResponse(
                    path=files[orig_idx][0],  # Original path
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return results  # type: ignore[return-value]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """
        여러 파일을 다운로드합니다. 효율성을 위해 백엔드별로 배칭합니다.

        경로를 대상 백엔드별로 그룹화하고, 각 백엔드의 download_files를
        해당 백엔드의 모든 경로와 함께 한 번 호출한 후, 결과를 원래 순서로 병합합니다.

        Args:
            paths: 다운로드할 파일 경로 목록.

        Returns:
            입력 경로당 하나씩 FileDownloadResponse 객체 목록.
            응답 순서는 입력 순서와 일치합니다.
        """
        # 결과 목록 사전 할당
        results: list[FileDownloadResponse | None] = [None] * len(paths)

        backend_batches: dict[BackendProtocol, list[tuple[int, str]]] = defaultdict(list)

        for idx, path in enumerate(paths):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path))

        # 각 백엔드의 배치 처리
        for backend, batch in backend_batches.items():
            # 백엔드 호출을 위한 데이터 추출
            indices, stripped_paths = zip(*batch, strict=False)

            # 백엔드를 모든 경로와 함께 한 번 호출
            batch_responses = backend.download_files(list(stripped_paths))

            # 원래 인덱스와 원래 경로로 응답 배치
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileDownloadResponse(
                    path=paths[orig_idx],  # 원래 경로
                    content=batch_responses[i].content if i < len(batch_responses) else None,
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return results  # type: ignore[return-value]

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """download_files의 비동기 버전입니다."""
        # 결과 목록 사전 할당
        results: list[FileDownloadResponse | None] = [None] * len(paths)

        backend_batches: dict[BackendProtocol, list[tuple[int, str]]] = defaultdict(list)

        for idx, path in enumerate(paths):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path))

        # Process each backend's batch
        for backend, batch in backend_batches.items():
            # Extract data for backend call
            indices, stripped_paths = zip(*batch, strict=False)

            # Call backend once with all its paths
            batch_responses = await backend.adownload_files(list(stripped_paths))

            # Place responses at original indices with original paths
            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileDownloadResponse(
                    path=paths[orig_idx],  # Original path
                    content=batch_responses[i].content if i < len(batch_responses) else None,
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return results  # type: ignore[return-value]
