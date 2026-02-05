"""
모듈명: _version.py
설명: Deep Agents 패키지 버전 정보

이 모듈은 패키지의 버전 문자열을 단일 소스로 관리합니다.
release-please 도구가 자동으로 버전을 업데이트합니다.

버전 형식: MAJOR.MINOR.PATCH (Semantic Versioning)
    - MAJOR: 호환되지 않는 API 변경
    - MINOR: 하위 호환 기능 추가
    - PATCH: 하위 호환 버그 수정

Note:
    x-release-please-version 주석은 CI/CD 파이프라인에서
    자동 버전 업데이트를 위해 사용됩니다. 삭제하지 마세요.
"""

# 패키지 버전 문자열
# - release-please가 릴리스 시 자동으로 업데이트
# - 형식: "MAJOR.MINOR.PATCH"
__version__ = "0.3.11"  # x-release-please-version
