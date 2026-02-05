"""
모듈명: skills.py
설명: 에이전트 스킬을 로드하고 시스템 프롬프트에 노출하는 미들웨어

이 모듈은 Anthropic의 에이전트 스킬 패턴을 점진적 공개(progressive disclosure)
방식으로 구현하며, 설정 가능한 소스를 통해 백엔드 저장소에서 스킬을 로드합니다.

## 아키텍처

스킬은 하나 이상의 **소스**에서 로드됩니다 - 소스는 백엔드에서 스킬이 구성된 경로입니다.
소스는 순서대로 로드되며, 동일한 이름의 스킬이 있을 경우 나중에 로드된 것이
이전 것을 덮어씁니다 (마지막 승리). 이를 통해 계층화가 가능합니다:
기본(base) -> 사용자(user) -> 프로젝트(project) -> 팀(team) 스킬.

미들웨어는 백엔드 API만 사용하므로 (직접적인 파일시스템 접근 없음),
다양한 저장소 백엔드에서 이식 가능합니다 (파일시스템, 상태, 원격 저장소 등).

StateBackend (임시/인메모리)의 경우, 팩토리 함수를 사용하세요:
```python
SkillsMiddleware(backend=lambda rt: StateBackend(rt), ...)
```

## 스킬 구조

각 스킬은 YAML 프론트매터가 포함된 SKILL.md 파일을 담은 디렉토리입니다:

```
/skills/user/web-research/
├── SKILL.md          # 필수: YAML 프론트매터 + 마크다운 지시사항
└── helper.py         # 선택: 지원 파일
```

SKILL.md 형식:
```markdown
---
name: web-research
description: 철저한 웹 리서치를 수행하는 구조화된 접근 방식
license: MIT
---

# 웹 리서치 스킬

## 사용 시기
- 사용자가 주제를 조사해달라고 요청할 때
...
```

## 스킬 메타데이터 (SkillMetadata)

Agent Skills 명세에 따라 YAML 프론트매터에서 파싱됩니다:
- `name`: 스킬 식별자 (최대 64자, 소문자 영숫자와 하이픈)
- `description`: 스킬의 기능 (최대 1024자)
- `path`: SKILL.md 파일의 백엔드 경로
- 선택사항: `license`, `compatibility`, `metadata`, `allowed_tools`

## 소스

소스는 단순히 백엔드의 스킬 디렉토리 경로입니다. 소스 이름은
경로의 마지막 구성요소에서 파생됩니다 (예: "/skills/user/" -> "user").

소스 예시:
```python
[
    "/skills/user/",
    "/skills/project/"
]
```

## 경로 규칙

모든 경로는 `PurePosixPath`를 통해 POSIX 규칙(슬래시)을 사용합니다:
- 백엔드 경로: "/skills/user/web-research/SKILL.md"
- 가상, 플랫폼 독립적
- 백엔드가 필요에 따라 플랫폼별 변환을 처리

## 사용법

```python
from deepagents.backends.state import StateBackend
from deepagents.middleware.skills import SkillsMiddleware

middleware = SkillsMiddleware(
    backend=my_backend,
    sources=[
        "/skills/base/",
        "/skills/user/",
        "/skills/project/",
    ],
)
```

주요 클래스:
- SkillMetadata: 스킬 메타데이터 TypedDict
- SkillsState: 스킬 미들웨어 상태
- SkillsMiddleware: 스킬 로드 및 시스템 프롬프트 주입 미들웨어
"""

from __future__ import annotations

import logging
import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Annotated

import yaml
from langchain.agents.middleware.types import PrivateStateAttr

if TYPE_CHECKING:
    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

from collections.abc import Awaitable, Callable
from typing import NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime

from deepagents.middleware._utils import append_to_system_message

logger = logging.getLogger(__name__)

# =============================================================================
# 상수 정의
# =============================================================================

# 보안: DoS 공격 방지를 위한 SKILL.md 파일 최대 크기 (10MB)
MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024

# Agent Skills 명세 제약 조건 (https://agentskills.io/specification)
MAX_SKILL_NAME_LENGTH = 64  # 스킬 이름 최대 길이
MAX_SKILL_DESCRIPTION_LENGTH = 1024  # 스킬 설명 최대 길이


# =============================================================================
# 타입 정의
# =============================================================================


class SkillMetadata(TypedDict):
    """
    Agent Skills 명세에 따른 스킬 메타데이터 (https://agentskills.io/specification)

    이 TypedDict는 SKILL.md 파일의 YAML 프론트매터에서 파싱된
    스킬 메타데이터를 표현합니다.

    필드:
        name: 스킬 식별자
            - 최대 64자
            - 소문자 영숫자와 하이픈만 허용 (a-z, 0-9, -)
            - 하이픈으로 시작하거나 끝날 수 없음
            - 연속 하이픈 불가
        description: 스킬의 기능 설명 (최대 1024자)
        path: SKILL.md 파일의 백엔드 경로
        license: 라이선스 이름 또는 번들된 라이선스 파일 참조
        compatibility: 환경 요구사항 (최대 500자)
        metadata: 추가 메타데이터를 위한 임의의 키-값 매핑
        allowed_tools: 사전 승인된 도구의 공백 구분 목록 (실험적)
    """

    name: str
    """스킬 식별자 (최대 64자, 소문자 영숫자와 하이픈)"""

    description: str
    """스킬의 기능 (최대 1024자)"""

    path: str
    """SKILL.md 파일의 경로"""

    license: str | None
    """라이선스 이름 또는 번들된 라이선스 파일 참조"""

    compatibility: str | None
    """환경 요구사항 (최대 500자)"""

    metadata: dict[str, str]
    """추가 메타데이터를 위한 임의의 키-값 매핑"""

    allowed_tools: list[str]
    """사전 승인된 도구의 공백 구분 목록 (실험적)"""


class SkillsState(AgentState):
    """
    스킬 미들웨어 상태

    AgentState를 확장하여 로드된 스킬 메타데이터를 포함합니다.
    skills_metadata는 PrivateStateAttr로 표시되어 부모 에이전트에게
    전파되지 않습니다.
    """

    skills_metadata: NotRequired[Annotated[list[SkillMetadata], PrivateStateAttr]]
    """설정된 소스에서 로드된 스킬 메타데이터 목록. 부모 에이전트에게 전파되지 않음."""


class SkillsStateUpdate(TypedDict):
    """
    스킬 미들웨어 상태 업데이트

    before_agent 훅에서 반환하여 상태에 스킬 메타데이터를 병합합니다.
    """

    skills_metadata: list[SkillMetadata]
    """상태에 병합할 로드된 스킬 메타데이터 목록"""


# =============================================================================
# 유틸리티 함수
# =============================================================================


def _validate_skill_name(name: str, directory_name: str) -> tuple[bool, str]:
    """
    Agent Skills 명세에 따라 스킬 이름을 검증합니다.

    명세에 따른 요구사항:
    - 최대 64자
    - 소문자 영숫자와 하이픈만 허용 (a-z, 0-9, -)
    - 하이픈으로 시작하거나 끝날 수 없음
    - 연속 하이픈 불가
    - 부모 디렉토리 이름과 일치해야 함

    인자:
        name: YAML 프론트매터의 스킬 이름
        directory_name: 부모 디렉토리 이름

    반환값:
        (is_valid, error_message) 튜플. 유효하면 에러 메시지는 빈 문자열.
    """
    if not name:
        return False, "name is required"
    if len(name) > MAX_SKILL_NAME_LENGTH:
        return False, "name exceeds 64 characters"
    # 패턴: 소문자 영숫자, 세그먼트 사이에 단일 하이픈, 시작/끝에 하이픈 없음
    if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", name):
        return False, "name must be lowercase alphanumeric with single hyphens only"
    if name != directory_name:
        return False, f"name '{name}' must match directory name '{directory_name}'"
    return True, ""


def _parse_skill_metadata(
    content: str,
    skill_path: str,
    directory_name: str,
) -> SkillMetadata | None:
    """
    SKILL.md 콘텐츠에서 YAML 프론트매터를 파싱합니다.

    콘텐츠 시작 부분의 --- 구분자로 구분된 YAML 프론트매터에서
    Agent Skills 명세에 따라 메타데이터를 추출합니다.

    인자:
        content: SKILL.md 파일의 콘텐츠
        skill_path: SKILL.md 파일의 경로 (에러 메시지 및 메타데이터용)
        directory_name: 스킬을 포함한 부모 디렉토리의 이름

    반환값:
        파싱이 성공하면 SkillMetadata, 파싱 실패 또는 검증 오류 시 None
    """
    # 파일 크기 검증 - DoS 공격 방지
    if len(content) > MAX_SKILL_FILE_SIZE:
        logger.warning("Skipping %s: content too large (%d bytes)", skill_path, len(content))
        return None

    # --- 구분자 사이의 YAML 프론트매터 매칭
    frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(frontmatter_pattern, content, re.DOTALL)

    if not match:
        logger.warning("Skipping %s: no valid YAML frontmatter found", skill_path)
        return None

    frontmatter_str = match.group(1)

    # 중첩 구조 지원을 위해 safe_load로 YAML 파싱
    try:
        frontmatter_data = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as e:
        logger.warning("Invalid YAML in %s: %s", skill_path, e)
        return None

    if not isinstance(frontmatter_data, dict):
        logger.warning("Skipping %s: frontmatter is not a mapping", skill_path)
        return None

    # 필수 필드 검증
    name = frontmatter_data.get("name")
    description = frontmatter_data.get("description")

    if not name or not description:
        logger.warning("Skipping %s: missing required 'name' or 'description'", skill_path)
        return None

    # 명세에 따른 이름 형식 검증 (하위 호환성을 위해 경고만 하고 로드는 계속)
    is_valid, error = _validate_skill_name(str(name), directory_name)
    if not is_valid:
        logger.warning(
            "Skill '%s' in %s does not follow Agent Skills specification: %s. Consider renaming for spec compliance.",
            name,
            skill_path,
            error,
        )

    # 명세에 따른 설명 길이 검증 (최대 1024자)
    description_str = str(description).strip()
    if len(description_str) > MAX_SKILL_DESCRIPTION_LENGTH:
        logger.warning(
            "Description exceeds %d characters in %s, truncating",
            MAX_SKILL_DESCRIPTION_LENGTH,
            skill_path,
        )
        description_str = description_str[:MAX_SKILL_DESCRIPTION_LENGTH]

    # 허용 도구 목록 파싱 (공백 구분)
    if frontmatter_data.get("allowed-tools"):
        allowed_tools = frontmatter_data.get("allowed-tools").split(" ")
    else:
        allowed_tools = []

    return SkillMetadata(
        name=str(name),
        description=description_str,
        path=skill_path,
        metadata=frontmatter_data.get("metadata", {}),
        license=frontmatter_data.get("license", "").strip() or None,
        compatibility=frontmatter_data.get("compatibility", "").strip() or None,
        allowed_tools=allowed_tools,
    )


def _list_skills(backend: BackendProtocol, source_path: str) -> list[SkillMetadata]:
    """
    백엔드 소스에서 모든 스킬을 나열합니다.

    백엔드에서 SKILL.md 파일을 포함하는 하위 디렉토리를 스캔하고,
    콘텐츠를 다운로드하고, YAML 프론트매터를 파싱하여 스킬 메타데이터를 반환합니다.

    예상 구조:
        source_path/
        ├── skill-name/
        │   ├── SKILL.md        # 필수
        │   └── helper.py       # 선택

    인자:
        backend: 파일 작업에 사용할 백엔드 인스턴스
        source_path: 백엔드의 스킬 디렉토리 경로

    반환값:
        성공적으로 파싱된 SKILL.md 파일의 스킬 메타데이터 목록
    """
    base_path = source_path

    skills: list[SkillMetadata] = []
    items = backend.ls_info(base_path)
    # 모든 스킬 디렉토리 찾기 (SKILL.md를 포함하는 디렉토리)
    skill_dirs = []
    for item in items:
        if not item.get("is_dir"):
            continue
        skill_dirs.append(item["path"])

    if not skill_dirs:
        return []

    # 각 스킬 디렉토리에 대해 SKILL.md 존재 여부 확인 및 다운로드
    skill_md_paths = []
    for skill_dir_path in skill_dirs:
        # 안전하고 표준화된 경로 작업을 위해 PurePosixPath로 SKILL.md 경로 구성
        skill_dir = PurePosixPath(skill_dir_path)
        skill_md_path = str(skill_dir / "SKILL.md")
        skill_md_paths.append((skill_dir_path, skill_md_path))

    paths_to_download = [skill_md_path for _, skill_md_path in skill_md_paths]
    responses = backend.download_files(paths_to_download)

    # 다운로드된 각 SKILL.md 파싱
    for (skill_dir_path, skill_md_path), response in zip(skill_md_paths, responses, strict=True):
        if response.error:
            # 스킬에 SKILL.md가 없으면 건너뛰기
            continue

        if response.content is None:
            logger.warning("Downloaded skill file %s has no content", skill_md_path)
            continue

        try:
            content = response.content.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.warning("Error decoding %s: %s", skill_md_path, e)
            continue

        # PurePosixPath를 사용하여 경로에서 디렉토리 이름 추출
        directory_name = PurePosixPath(skill_dir_path).name

        # 메타데이터 파싱
        skill_metadata = _parse_skill_metadata(
            content=content,
            skill_path=skill_md_path,
            directory_name=directory_name,
        )
        if skill_metadata:
            skills.append(skill_metadata)

    return skills


async def _alist_skills(backend: BackendProtocol, source_path: str) -> list[SkillMetadata]:
    """
    백엔드 소스에서 모든 스킬을 나열합니다 (비동기 버전).

    백엔드에서 SKILL.md 파일을 포함하는 하위 디렉토리를 스캔하고,
    콘텐츠를 다운로드하고, YAML 프론트매터를 파싱하여 스킬 메타데이터를 반환합니다.

    예상 구조:
        source_path/
        ├── skill-name/
        │   ├── SKILL.md        # 필수
        │   └── helper.py       # 선택

    인자:
        backend: 파일 작업에 사용할 백엔드 인스턴스
        source_path: 백엔드의 스킬 디렉토리 경로

    반환값:
        성공적으로 파싱된 SKILL.md 파일의 스킬 메타데이터 목록
    """
    base_path = source_path

    skills: list[SkillMetadata] = []
    items = await backend.als_info(base_path)
    # 모든 스킬 디렉토리 찾기 (SKILL.md를 포함하는 디렉토리)
    skill_dirs = []
    for item in items:
        if not item.get("is_dir"):
            continue
        skill_dirs.append(item["path"])

    if not skill_dirs:
        return []

    # 각 스킬 디렉토리에 대해 SKILL.md 존재 여부 확인 및 다운로드
    skill_md_paths = []
    for skill_dir_path in skill_dirs:
        # 안전하고 표준화된 경로 작업을 위해 PurePosixPath로 SKILL.md 경로 구성
        skill_dir = PurePosixPath(skill_dir_path)
        skill_md_path = str(skill_dir / "SKILL.md")
        skill_md_paths.append((skill_dir_path, skill_md_path))

    paths_to_download = [skill_md_path for _, skill_md_path in skill_md_paths]
    responses = await backend.adownload_files(paths_to_download)

    # 다운로드된 각 SKILL.md 파싱
    for (skill_dir_path, skill_md_path), response in zip(skill_md_paths, responses, strict=True):
        if response.error:
            # 스킬에 SKILL.md가 없으면 건너뛰기
            continue

        if response.content is None:
            logger.warning("Downloaded skill file %s has no content", skill_md_path)
            continue

        try:
            content = response.content.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.warning("Error decoding %s: %s", skill_md_path, e)
            continue

        # PurePosixPath를 사용하여 경로에서 디렉토리 이름 추출
        directory_name = PurePosixPath(skill_dir_path).name

        # 메타데이터 파싱
        skill_metadata = _parse_skill_metadata(
            content=content,
            skill_path=skill_md_path,
            directory_name=directory_name,
        )
        if skill_metadata:
            skills.append(skill_metadata)

    return skills


# =============================================================================
# 시스템 프롬프트 템플릿
# =============================================================================

# 스킬 시스템 설명 및 사용법을 담은 시스템 프롬프트 템플릿
# {skills_locations}와 {skills_list}는 런타임에 실제 값으로 치환됨
SKILLS_SYSTEM_PROMPT = """

## Skills System

You have access to a skills library that provides specialized capabilities and domain knowledge.

{skills_locations}

**Available Skills:**

{skills_list}

**How to Use Skills (Progressive Disclosure):**

Skills follow a **progressive disclosure** pattern - you see their name and description above, but only read full instructions when needed:

1. **Recognize when a skill applies**: Check if the user's task matches a skill's description
2. **Read the skill's full instructions**: Use the path shown in the skill list above
3. **Follow the skill's instructions**: SKILL.md contains step-by-step workflows, best practices, and examples
4. **Access supporting files**: Skills may include helper scripts, configs, or reference docs - use absolute paths

**When to Use Skills:**
- User's request matches a skill's domain (e.g., "research X" -> web-research skill)
- You need specialized knowledge or structured workflows
- A skill provides proven patterns for complex tasks

**Executing Skill Scripts:**
Skills may contain Python scripts or other executable files. Always use absolute paths from the skill list.

**Example Workflow:**

User: "Can you research the latest developments in quantum computing?"

1. Check available skills -> See "web-research" skill with its path
2. Read the skill using the path shown
3. Follow the skill's research workflow (search -> organize -> synthesize)
4. Use any helper scripts with absolute paths

Remember: Skills make you more capable and consistent. When in doubt, check if a skill exists for the task!
"""


# =============================================================================
# 메인 미들웨어 클래스
# =============================================================================


class SkillsMiddleware(AgentMiddleware):
    """
    에이전트 스킬을 로드하고 시스템 프롬프트에 노출하는 미들웨어

    백엔드 소스에서 스킬을 로드하고 점진적 공개(progressive disclosure) 방식으로
    시스템 프롬프트에 주입합니다 (메타데이터 먼저, 필요시 전체 콘텐츠).

    스킬은 소스 순서대로 로드되며, 나중에 로드된 소스가 이전 것을 덮어씁니다.

    예시:
        ```python
        from deepagents.backends.filesystem import FilesystemBackend

        backend = FilesystemBackend(root_dir="/path/to/skills")
        middleware = SkillsMiddleware(
            backend=backend,
            sources=[
                "/path/to/skills/user/",
                "/path/to/skills/project/",
            ],
        )
        ```

    속성:
        state_schema: 상태 스키마 (SkillsState)
        sources: 스킬 소스 경로 목록
        system_prompt_template: 시스템 프롬프트 템플릿

    인자:
        backend: 파일 작업을 위한 백엔드 인스턴스
        sources: 스킬 소스 경로 목록. 소스 이름은 경로의 마지막 구성요소에서 파생됨.
    """

    state_schema = SkillsState

    def __init__(self, *, backend: BACKEND_TYPES, sources: list[str]) -> None:
        """
        스킬 미들웨어를 초기화합니다.

        인자:
            backend: 백엔드 인스턴스 또는 런타임을 받아 백엔드를 반환하는 팩토리 함수.
                     StateBackend의 경우 팩토리 사용: `lambda rt: StateBackend(rt)`
            sources: 스킬 소스 경로 목록 (예: ["/skills/user/", "/skills/project/"]).
        """
        self._backend = backend
        self.sources = sources
        self.system_prompt_template = SKILLS_SYSTEM_PROMPT

    def _get_backend(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> BackendProtocol:
        """
        인스턴스 또는 팩토리에서 백엔드를 해석합니다.

        인자:
            state: 현재 에이전트 상태.
            runtime: 팩토리 함수를 위한 런타임 컨텍스트.
            config: 백엔드 팩토리에 전달할 Runnable 설정.

        반환값:
            해석된 백엔드 인스턴스
        """
        if callable(self._backend):
            # 백엔드 팩토리 해석을 위한 인위적인 도구 런타임 생성
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            backend = self._backend(tool_runtime)
            if backend is None:
                raise AssertionError("SkillsMiddleware requires a valid backend instance")
            return backend

        return self._backend

    def _format_skills_locations(self) -> str:
        """시스템 프롬프트에 표시할 스킬 위치를 포맷합니다."""
        locations = []
        for i, source_path in enumerate(self.sources):
            name = PurePosixPath(source_path.rstrip("/")).name.capitalize()
            # 마지막 소스가 가장 높은 우선순위를 가짐
            suffix = " (higher priority)" if i == len(self.sources) - 1 else ""
            locations.append(f"**{name} Skills**: `{source_path}`{suffix}")
        return "\n".join(locations)

    def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
        """시스템 프롬프트에 표시할 스킬 메타데이터를 포맷합니다."""
        if not skills:
            paths = [f"{source_path}" for source_path in self.sources]
            return f"(No skills available yet. You can create skills in {' or '.join(paths)})"

        lines = []
        for skill in skills:
            lines.append(f"- **{skill['name']}**: {skill['description']}")
            if skill["allowed_tools"]:
                lines.append(f"  -> Allowed tools: {', '.join(skill['allowed_tools'])}")
            lines.append(f"  -> Read `{skill['path']}` for full instructions")

        return "\n".join(lines)

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """
        모델 요청의 시스템 메시지에 스킬 문서를 주입합니다.

        인자:
            request: 수정할 모델 요청

        반환값:
            시스템 메시지에 스킬 문서가 주입된 새 모델 요청
        """
        skills_metadata = request.state.get("skills_metadata", [])
        skills_locations = self._format_skills_locations()
        skills_list = self._format_skills_list(skills_metadata)

        skills_section = self.system_prompt_template.format(
            skills_locations=skills_locations,
            skills_list=skills_list,
        )

        new_system_message = append_to_system_message(request.system_message, skills_section)

        return request.override(system_message=new_system_message)

    def before_agent(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> SkillsStateUpdate | None:
        """
        에이전트 실행 전에 스킬 메타데이터를 로드합니다 (동기).

        설정된 모든 소스에서 사용 가능한 스킬을 발견하기 위해
        각 에이전트 상호작용 전에 실행됩니다.
        변경사항을 캡처하기 위해 매 호출마다 재로드합니다.

        스킬은 소스 순서대로 로드되며, 동일한 이름의 스킬이 있을 경우
        나중에 로드된 소스가 이전 것을 덮어씁니다 (마지막 승리).

        인자:
            state: 현재 에이전트 상태.
            runtime: 런타임 컨텍스트.
            config: Runnable 설정.

        반환값:
            `skills_metadata`가 채워진 상태 업데이트, 이미 존재하면 `None`
        """
        # skills_metadata가 상태에 이미 존재하면 건너뛰기 (비어있어도)
        if "skills_metadata" in state:
            return None

        # 백엔드 해석 (직접 인스턴스와 팩토리 함수 모두 지원)
        backend = self._get_backend(state, runtime, config)
        all_skills: dict[str, SkillMetadata] = {}

        # 각 소스에서 순서대로 스킬 로드
        # 나중 소스가 이전 것을 덮어씀 (마지막 승리)
        for source_path in self.sources:
            source_skills = _list_skills(backend, source_path)
            for skill in source_skills:
                all_skills[skill["name"]] = skill

        skills = list(all_skills.values())
        return SkillsStateUpdate(skills_metadata=skills)

    async def abefore_agent(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> SkillsStateUpdate | None:
        """
        에이전트 실행 전에 스킬 메타데이터를 로드합니다 (비동기).

        설정된 모든 소스에서 사용 가능한 스킬을 발견하기 위해
        각 에이전트 상호작용 전에 실행됩니다.
        변경사항을 캡처하기 위해 매 호출마다 재로드합니다.

        스킬은 소스 순서대로 로드되며, 동일한 이름의 스킬이 있을 경우
        나중에 로드된 소스가 이전 것을 덮어씁니다 (마지막 승리).

        인자:
            state: 현재 에이전트 상태.
            runtime: 런타임 컨텍스트.
            config: Runnable 설정.

        반환값:
            `skills_metadata`가 채워진 상태 업데이트, 이미 존재하면 `None`
        """
        # skills_metadata가 상태에 이미 존재하면 건너뛰기 (비어있어도)
        if "skills_metadata" in state:
            return None

        # 백엔드 해석 (직접 인스턴스와 팩토리 함수 모두 지원)
        backend = self._get_backend(state, runtime, config)
        all_skills: dict[str, SkillMetadata] = {}

        # 각 소스에서 순서대로 스킬 로드
        # 나중 소스가 이전 것을 덮어씀 (마지막 승리)
        for source_path in self.sources:
            source_skills = await _alist_skills(backend, source_path)
            for skill in source_skills:
                all_skills[skill["name"]] = skill

        skills = list(all_skills.values())
        return SkillsStateUpdate(skills_metadata=skills)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """
        시스템 프롬프트에 스킬 문서를 주입합니다.

        인자:
            request: 처리 중인 모델 요청
            handler: 수정된 요청으로 호출할 핸들러 함수

        반환값:
            핸들러의 모델 응답
        """
        modified_request = self.modify_request(request)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """
        시스템 프롬프트에 스킬 문서를 주입합니다 (비동기 버전).

        인자:
            request: 처리 중인 모델 요청
            handler: 수정된 요청으로 호출할 비동기 핸들러 함수

        반환값:
            핸들러의 모델 응답
        """
        modified_request = self.modify_request(request)
        return await handler(modified_request)


__all__ = ["SkillMetadata", "SkillsMiddleware"]
