"""
모듈명: __main__.py
설명: ACP DeepAgent의 진입점

주요 기능:
- main(): 지정된 루트 디렉토리로 ACP DeepAgent 실행
  - --root-dir: 에이전트가 접근 가능한 루트 디렉토리 (기본값: 현재 작업 디렉토리)

의존성:
- deepagents_acp.agent: 에이전트 실행 함수
"""

import argparse
import asyncio
import os

from deepagents_acp.agent import run_agent


def main():
    parser = argparse.ArgumentParser(description="Run ACP DeepAgent with specified root directory")
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="Root directory accessible to the agent (default: current working directory)",
    )
    args = parser.parse_args()
    root_dir = args.root_dir if args.root_dir else os.getcwd()
    asyncio.run(run_agent(root_dir))


if __name__ == "__main__":
    main()
