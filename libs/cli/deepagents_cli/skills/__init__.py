"""
모듈명: skills/__init__.py
설명: deepagents CLI용 스킬 모듈

공개 API:
- execute_skills_command: 스킬 서브커맨드 실행 (list/create/info)
- setup_skills_parser: 스킬 명령의 argparse 설정

기타 컴포넌트는 내부 구현 세부 사항입니다.
"""

from deepagents_cli.skills.commands import (
    execute_skills_command,
    setup_skills_parser,
)

__all__ = [
    "execute_skills_command",
    "setup_skills_parser",
]
