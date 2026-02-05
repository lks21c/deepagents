"""
모듈명: utils.py
설명: ACP 콘텐츠 블록을 LangChain 형식으로 변환하는 유틸리티

주요 기능:
- convert_text_block_to_content_blocks(): 텍스트 블록 변환
- convert_image_block_to_content_blocks(): 이미지 블록 변환 (base64 데이터 URI)
- convert_audio_block_to_content_blocks(): 오디오 블록 변환 (미지원)
- convert_resource_block_to_content_blocks(): 리소스 참조 블록 변환
- convert_embedded_resource_block_to_content_blocks(): 임베디드 리소스 블록 변환

의존성:
- acp.schema: ACP 콘텐츠 블록 스키마
"""

from acp.schema import (
    AudioContentBlock,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
)


def convert_text_block_to_content_blocks(block: TextContentBlock):
    return [{"type": "text", "text": block.text}]


def convert_image_block_to_content_blocks(block: ImageContentBlock):
    # Image blocks contain visual data
    # Primary case: inline base64 data (data is already a base64 string)
    if block.data:
        data_uri = f"data:{block.mime_type};base64,{block.data}"
        return [{"type": "image_url", "image_url": {"url": data_uri}}]

    # No data available
    return [{"type": "text", "text": "[Image: no data available]"}]


def convert_audio_block_to_content_blocks(block: AudioContentBlock):
    raise Exception("Audio is not currently supported.")


def convert_resource_block_to_content_blocks(block: ResourceContentBlock, *, root_dir: str):
    # Resource blocks reference external resources
    resource_text = f"[Resource: {block.name}"
    if block.uri:
        # Truncate root_dir from path while preserving file:// prefix
        uri = block.uri
        has_file_prefix = uri.startswith("file://")
        if has_file_prefix:
            path = uri[7:]  # Remove "file://" temporarily
        else:
            path = uri

        # Remove root_dir prefix to get path relative to agent's working directory
        if path.startswith(root_dir):
            path = path[len(root_dir) :].lstrip("/")

        # Restore file:// prefix if it was present
        uri = f"file://{path}" if has_file_prefix else path
        resource_text += f"\nURI: {uri}"
    if block.description:
        resource_text += f"\nDescription: {block.description}"
    if block.mime_type:
        resource_text += f"\nMIME type: {block.mime_type}"
    resource_text += "]"
    return [{"type": "text", "text": resource_text}]


def convert_embedded_resource_block_to_content_blocks(
    block: EmbeddedResourceContentBlock,
) -> list[dict]:
    # Embedded resource blocks contain the resource data inline
    resource = block.resource
    if hasattr(resource, "text"):
        mime_type = getattr(resource, "mime_type", "application/text")
        return [{"type": "text", "text": f"[Embedded {mime_type} resource: {resource.text}"}]
    elif hasattr(resource, "blob"):
        mime_type = getattr(resource, "mime_type", "application/octet-stream")
        data_uri = f"data:{mime_type};base64,{resource.blob}"
        return [
            {
                "type": "text",
                "text": f"[Embedded resource: {data_uri}]",
            }
        ]
    else:
        raise Exception(
            "Could not parse embedded resource block. "
            "Block expected either a `text` or `blob` property."
        )
