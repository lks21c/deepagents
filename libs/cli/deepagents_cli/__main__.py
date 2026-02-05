"""
모듈명: __main__.py
설명: CLI를 python -m deepagents.cli로 실행할 수 있게 하는 진입점

이 모듈은 패키지를 직접 실행할 때 호출됩니다:
    python -m deepagents_cli

사용 예시:
    python -m deepagents_cli
    python -m deepagents_cli --help
    python -m deepagents_cli --model claude-sonnet-4-5-20250929
"""

from deepagents_cli.main import cli_main

if __name__ == "__main__":
    cli_main()
