"""
모듈명: tracing.py
설명: Harbor Deep Agents용 LangSmith 통합

주요 기능:
- create_example_id_from_instruction(): 지시사항 문자열에서 결정론적 UUID 생성
  - 공백 정규화 및 SHA-256 해싱
  - LangSmith 호환 UUID 반환

의존성:
- hashlib: SHA-256 해싱
- uuid: UUID 생성
"""

import hashlib
import uuid


def create_example_id_from_instruction(instruction: str, seed: int = 42) -> str:
    """Create a deterministic UUID from an instruction string.

    Normalizes the instruction by stripping whitespace and creating a
    SHA-256 hash, then converting to a UUID for LangSmith compatibility.

    Args:
        instruction: The task instruction string to hash
        seed: Integer seed to avoid collisions with existing examples

    Returns:
        A UUID string generated from the hash of the normalized instruction
    """
    # Normalize the instruction: strip leading/trailing whitespace
    normalized = instruction.strip()

    # Prepend seed as bytes to the instruction for hashing
    seeded_data = seed.to_bytes(8, byteorder="big") + normalized.encode("utf-8")

    # Create SHA-256 hash of the seeded instruction
    hash_bytes = hashlib.sha256(seeded_data).digest()

    # Use first 16 bytes to create a UUID
    example_uuid = uuid.UUID(bytes=hash_bytes[:16])

    return str(example_uuid)
