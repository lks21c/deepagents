"""
모듈명: _utils.py
설명: 미들웨어 모듈에서 공통으로 사용되는 유틸리티 함수들

이 모듈은 여러 미들웨어에서 재사용되는 헬퍼 함수들을 제공합니다.
주로 시스템 메시지 조작과 관련된 기능을 포함합니다.

주요 함수:
    - append_to_system_message: 시스템 메시지에 텍스트를 추가
"""

from langchain_core.messages import SystemMessage


def append_to_system_message(
    system_message: SystemMessage | None,
    text: str,
) -> SystemMessage:
    """시스템 메시지에 텍스트를 추가합니다.

    기존 시스템 메시지가 있으면 새로운 텍스트를 끝에 추가하고,
    없으면 새로운 시스템 메시지를 생성합니다. 텍스트는 콘텐츠 블록
    형식으로 추가되어 멀티모달 메시지 구조를 유지합니다.

    Args:
        system_message: 기존 시스템 메시지. None이면 새로 생성합니다.
        text: 시스템 메시지에 추가할 텍스트.

    Returns:
        텍스트가 추가된 새로운 SystemMessage 객체.

    사용 예시:
        ```python
        # 기존 시스템 메시지에 추가
        existing = SystemMessage(content="기존 지시사항")
        updated = append_to_system_message(existing, "추가 지시사항")
        # 결과: "기존 지시사항\n\n추가 지시사항"

        # None에서 새로 생성
        new_msg = append_to_system_message(None, "새로운 지시사항")
        # 결과: "새로운 지시사항"
        ```
    """
    # 기존 시스템 메시지의 콘텐츠 블록을 복사하거나 빈 리스트 생성
    new_content: list[str | dict[str, str]] = list(system_message.content_blocks) if system_message else []

    # 기존 콘텐츠가 있으면 구분을 위해 빈 줄 2개 추가
    if new_content:
        text = f"\n\n{text}"

    # 새 텍스트를 콘텐츠 블록 형식으로 추가
    new_content.append({"type": "text", "text": text})

    return SystemMessage(content=new_content)
