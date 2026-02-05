"""
모듈명: local_shell.py
설명: 무제한 로컬 셸 실행 기능을 갖춘 파일시스템 백엔드

이 백엔드는 FilesystemBackend를 확장하여 로컬 호스트 시스템에서
셸 명령 실행 기능을 추가합니다. 샌드박싱이나 격리가 전혀 없으며,
모든 작업이 호스트 머신에서 전체 시스템 접근 권한으로 직접 실행됩니다.

주요 클래스:
    - LocalShellBackend: 파일시스템 접근과 셸 실행 기능을 결합한 백엔드

⚠️ 보안 경고:
    이 백엔드는 에이전트에게 파일시스템과 셸 모두에 대한
    무제한 접근을 부여합니다. 신뢰할 수 있는 환경에서만 사용하세요.

사용 시나리오:
    - 로컬 개발 CLI (코딩 어시스턴트, 개발 도구)
    - 에이전트 코드를 신뢰하는 개인 개발 환경
    - 적절한 시크릿 관리가 갖춰진 CI/CD 파이프라인

부적절한 사용:
    - 프로덕션 환경 (웹 서버, API, 멀티 테넌트 시스템)
    - 신뢰할 수 없는 사용자 입력 처리 또는 비신뢰 코드 실행
"""

from __future__ import annotations

import os
import subprocess
import uuid
from typing import TYPE_CHECKING

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

if TYPE_CHECKING:
    from pathlib import Path


class LocalShellBackend(FilesystemBackend, SandboxBackendProtocol):
    """무제한 로컬 셸 명령 실행 기능을 갖춘 파일시스템 백엔드.

    이 백엔드는 FilesystemBackend를 확장하여 셸 명령 실행 기능을 추가합니다.
    명령은 샌드박싱, 프로세스 격리, 보안 제한 없이 호스트 시스템에서
    직접 실행됩니다.

    ⚠️ 보안 경고:
        이 백엔드는 에이전트에게 로컬 머신에 대한 직접적인 파일시스템 접근과
        무제한 셸 실행 권한을 모두 부여합니다. 극도의 주의를 기울여
        적절한 환경에서만 사용하세요.

        적절한 사용 사례:
            - 로컬 개발 CLI (코딩 어시스턴트, 개발 도구)
            - 에이전트 코드를 신뢰하는 개인 개발 환경
            - 적절한 시크릿 관리가 갖춰진 CI/CD 파이프라인

        부적절한 사용 사례:
            - 프로덕션 환경 (예: 웹 서버, API, 멀티 테넌트 시스템)
            - 신뢰할 수 없는 사용자 입력 처리 또는 비신뢰 코드 실행

        프로덕션 환경에서는 StateBackend, StoreBackend 또는
        BaseSandbox를 확장하여 사용하세요.

        보안 위험:
            - 에이전트가 사용자 권한으로 임의의 셸 명령 실행 가능
            - 에이전트가 시크릿을 포함한 모든 접근 가능한 파일 읽기 가능
              (API 키, 자격증명, .env 파일, SSH 키 등)
            - 네트워크 도구와 결합 시 SSRF 공격으로 시크릿 유출 가능
            - 파일 수정과 명령 실행은 영구적이며 되돌릴 수 없음
            - 에이전트가 패키지 설치, 시스템 파일 수정, 프로세스 생성 가능
            - 프로세스 격리 없음 - 명령이 호스트 시스템에서 직접 실행
            - 리소스 제한 없음 - 명령이 무제한 CPU, 메모리, 디스크 사용 가능

        권장 보호 조치:
            셸 접근이 무제한이며 파일시스템 제한을 우회할 수 있으므로:

            1. Human-in-the-Loop (HITL) 미들웨어 활성화하여 실행 전
               모든 작업을 검토하고 승인. 이 백엔드 사용 시 주요 보호 장치로
               강력히 권장됨.
            2. 전용 개발 환경에서만 실행 - 공유 또는 프로덕션 시스템에서 절대 금지
            3. 신뢰할 수 없는 사용자에게 노출하거나 비신뢰 코드 실행 금지
            4. 코드 실행이 필요한 프로덕션 환경에서는 BaseSandbox를 확장하여
               적절히 격리된 백엔드 생성 (Docker 컨테이너, VM 등)

        참고:
            virtual_mode=True와 경로 기반 제한은 셸 접근이 활성화된 상태에서는
            보안을 제공하지 않습니다. 명령이 시스템의 모든 경로에 접근할 수 있기 때문입니다.

    사용 예시:
        ```python
        from deepagents.backends import LocalShellBackend

        # 명시적 환경 변수로 백엔드 생성
        backend = LocalShellBackend(root_dir="/home/user/project", env={"PATH": "/usr/bin:/bin"})

        # 셸 명령 실행 (호스트에서 직접 실행)
        result = backend.execute("ls -la")
        print(result.output)
        print(result.exit_code)

        # 파일시스템 작업 사용 (FilesystemBackend에서 상속)
        content = backend.read("/README.md")
        backend.write("/output.txt", "Hello world")

        # 모든 환경 변수 상속
        backend = LocalShellBackend(root_dir="/home/user/project", inherit_env=True)
        ```
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        *,
        virtual_mode: bool = False,
        timeout: float = 120.0,
        max_output_bytes: int = 100_000,
        env: dict[str, str] | None = None,
        inherit_env: bool = False,
    ) -> None:
        """파일시스템 접근 기능을 갖춘 로컬 셸 백엔드를 초기화합니다.

        인자:
            root_dir: 파일시스템 작업과 셸 명령 모두의 작업 디렉토리.

                - 제공되지 않으면 현재 작업 디렉토리가 기본값
                - 셸 명령은 이 디렉토리를 작업 디렉토리로 사용하여 실행
                - virtual_mode=False (기본값): 경로가 그대로 사용됨. 에이전트가
                  절대 경로나 .. 시퀀스로 모든 파일에 접근 가능
                - virtual_mode=True: 파일시스템 작업의 가상 루트로 동작.
                  CompositeBackend와 함께 사용하여 여러 백엔드 구현에
                  파일 작업 라우팅 지원. 참고: 셸 명령은 제한하지 않음

            virtual_mode: 파일시스템 작업에 대한 가상 경로 모드 활성화.

                True일 때, root_dir을 가상 루트 파일시스템으로 취급.
                모든 경로가 root_dir 기준 상대 경로로 해석됨
                (예: /file.txt가 {root_dir}/file.txt로 매핑).
                경로 탐색 (.., ~)은 차단됨.

                주요 사용 사례: CompositeBackend와 함께 사용. CompositeBackend는
                다른 경로 접두사를 다른 백엔드로 라우팅. 가상 모드를 통해
                CompositeBackend가 라우트 접두사를 제거하고 정규화된 경로를
                각 백엔드에 전달하여 여러 백엔드 구현에서 파일 작업이
                올바르게 동작하도록 함.

                중요: 이것은 파일시스템 작업에만 영향. execute()를 통해 실행되는
                셸 명령은 제한되지 않으며 모든 경로에 접근 가능.

            timeout: 셸 명령 실행 대기 최대 시간(초).
                이 타임아웃을 초과하는 명령은 종료됨. 기본값 120초.

            max_output_bytes: 명령 출력에서 캡처할 최대 바이트 수.
                이 제한을 초과하는 출력은 잘림. 기본값 100,000바이트.

            env: 셸 명령의 환경 변수. None이면 빈 환경으로 시작
                (inherit_env=True가 아닌 경우).

            inherit_env: 부모 프로세스의 환경 변수를 상속할지 여부.
                False (기본값)이면 env 딕셔너리의 변수만 사용 가능.
                True이면 모든 os.environ 변수를 상속하고 env 오버라이드 적용.
        """
        # 부모 FilesystemBackend 초기화
        super().__init__(
            root_dir=root_dir,
            virtual_mode=virtual_mode,
            max_file_size_mb=10,
        )

        # 실행 파라미터 저장
        self._timeout = timeout
        self._max_output_bytes = max_output_bytes

        # inherit_env 설정에 따라 환경 변수 구성
        if inherit_env:
            # 부모 프로세스의 환경 변수를 모두 복사
            self._env = os.environ.copy()
            if env is not None:
                # 사용자 지정 환경 변수로 오버라이드
                self._env.update(env)
        else:
            # 빈 환경에서 시작하거나 명시된 환경만 사용
            self._env = env if env is not None else {}

        # 고유 샌드박스 ID 생성 (local-{8자리 랜덤 hex} 형식)
        self._sandbox_id = f"local-{uuid.uuid4().hex[:8]}"

    @property
    def id(self) -> str:
        """이 백엔드 인스턴스의 고유 식별자.

        반환값:
            "local-{랜덤_hex}" 형식의 문자열 식별자.
        """
        return self._sandbox_id

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        r"""호스트 시스템에서 셸 명령을 직접 실행합니다.

        ⚠️ 위험: 무제한 실행
            명령은 shell=True 옵션의 subprocess.run()을 사용하여
            호스트 시스템에서 직접 실행됩니다. 샌드박싱, 격리, 보안 제한이
            전혀 없습니다. 명령은 사용자의 전체 권한으로 실행되며:

            - 파일시스템의 모든 파일 접근 가능 (virtual_mode와 무관)
            - 모든 프로그램이나 스크립트 실행 가능
            - 네트워크 연결 생성 가능
            - 시스템 설정 수정 가능
            - 추가 프로세스 생성 가능
            - 패키지 설치 또는 의존성 수정 가능

            이 메서드를 사용할 때는 항상 Human-in-the-Loop (HITL) 미들웨어를 사용하세요.

        명령은 시스템 셸 (/bin/sh 또는 동등)을 사용하여 실행되며,
        작업 디렉토리는 백엔드의 root_dir로 설정됩니다.
        stdout과 stderr는 단일 출력 스트림으로 결합됩니다.

        인자:
            command: 실행할 셸 명령 문자열.
                예: "python script.py", "ls -la", "grep pattern file.txt"

                보안 주의: 이 문자열은 셸에 직접 전달됩니다. 에이전트가
                파이프, 리다이렉트, 명령 치환 등을 포함한 임의의 명령을
                실행할 수 있습니다.

        반환값:
            다음을 포함하는 ExecuteResponse:
                - output: 결합된 stdout과 stderr (stderr 라인은 [stderr] 접두사 포함)
                - exit_code: 프로세스 종료 코드 (성공 시 0, 실패 시 0이 아닌 값)
                - truncated: 크기 제한으로 출력이 잘렸으면 True

        사용 예시:
            ```python
            # 간단한 명령 실행
            result = backend.execute("echo hello")
            assert result.output == "hello\\n"
            assert result.exit_code == 0

            # 오류 처리
            result = backend.execute("cat nonexistent.txt")
            assert result.exit_code != 0
            assert "[stderr]" in result.output

            # 잘림 확인
            result = backend.execute("cat huge_file.txt")
            if result.truncated:
                print("출력이 잘렸습니다")

            # 명령은 root_dir에서 실행되지만 모든 경로에 접근 가능
            result = backend.execute("cat /etc/passwd")  # 시스템 파일 읽기 가능!
            ```
        """
        # 명령 유효성 검사: 빈 문자열이나 문자열이 아닌 입력 거부
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        try:
            # subprocess.run으로 셸 명령 실행
            # shell=True는 의도적: LLM 제어 셸 실행을 위해 설계됨
            result = subprocess.run(  # noqa: S602
                command,
                check=False,  # 실패해도 예외 발생 안 함, 종료 코드로 확인
                shell=True,  # 의도적: LLM 제어 셸 실행을 위해 설계됨
                capture_output=True,  # stdout과 stderr 모두 캡처
                text=True,  # 출력을 문자열로 (바이트가 아닌)
                timeout=self._timeout,  # 설정된 타임아웃 적용
                env=self._env,  # 구성된 환경 변수 사용
                cwd=str(self.cwd),  # FilesystemBackend의 root_dir을 작업 디렉토리로 사용
            )

            # stdout과 stderr를 결합
            # 각 stderr 라인에 [stderr] 접두사를 붙여 명확히 구분
            # 예: "hello\n[stderr] error: file not found"  # noqa: ERA001
            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                stderr_lines = result.stderr.strip().split("\n")
                output_parts.extend(f"[stderr] {line}" for line in stderr_lines)

            # 출력이 없으면 "<no output>" 표시
            output = "\n".join(output_parts) if output_parts else "<no output>"

            # 잘림 확인: 출력이 최대 바이트를 초과하는지 체크
            truncated = False
            if len(output) > self._max_output_bytes:
                output = output[: self._max_output_bytes]
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
                truncated = True

            # 종료 코드가 0이 아니면 종료 코드 정보 추가
            if result.returncode != 0:
                output = f"{output.rstrip()}\n\nExit code: {result.returncode}"

            return ExecuteResponse(
                output=output,
                exit_code=result.returncode,
                truncated=truncated,
            )

        except subprocess.TimeoutExpired:
            # 명령이 타임아웃을 초과한 경우
            return ExecuteResponse(
                output=f"Error: Command timed out after {self._timeout:.1f} seconds.",
                exit_code=124,  # 표준 타임아웃 종료 코드
                truncated=False,
            )
        except Exception as e:  # noqa: BLE001
            # 광범위한 예외 캐치는 의도적: 모든 실행 오류를 캐치하고
            # 예외를 전파하지 않고 일관된 ExecuteResponse를 반환
            return ExecuteResponse(
                output=f"Error executing command: {e}",
                exit_code=1,
                truncated=False,
            )


__all__ = ["LocalShellBackend"]
