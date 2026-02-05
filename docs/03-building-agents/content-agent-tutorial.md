# 콘텐츠 에이전트 튜토리얼

> 블로그 포스트, 소셜 미디어 콘텐츠를 생성하는 에이전트를 구축합니다.

## 개요

이 튜토리얼에서는 다음 기능을 가진 콘텐츠 에이전트를 구축합니다:

- 블로그 포스트 작성
- 소셜 미디어 콘텐츠 생성
- 리서치 서브에이전트와 협업
- 이미지 생성 통합
- 파일 기반 설정 (AGENTS.md, skills, subagents.yaml)

**소스 참조**: `examples/content-builder-agent/content_writer.py`

## 사전 요구사항

```bash
pip install deepagents tavily-python pyyaml
```

## 프로젝트 구조

```
content_agent/
├── content_writer.py    # 메인 에이전트
├── AGENTS.md            # 브랜드 보이스 & 스타일 가이드
├── subagents.yaml       # 서브에이전트 정의
└── skills/
    ├── blog-writing/
    │   └── SKILL.md
    └── social-media/
        └── SKILL.md
```

---

## Step 1: AGENTS.md 작성

브랜드 보이스와 스타일 가이드를 정의합니다.

### AGENTS.md

```markdown
# Content Writer Agent

당신은 전문 콘텐츠 라이터입니다.

## Brand Voice

### 톤 & 스타일
- **친근하면서 전문적**: 기술 용어를 쉽게 설명
- **실용적**: 독자가 바로 적용할 수 있는 내용
- **간결함**: 불필요한 단어 제거

### 금지 사항
- 과도한 전문 용어 남용
- 클릭베이트 제목
- 근거 없는 주장

## Content Guidelines

### 블로그 포스트
- 길이: 1500-2500 단어
- 구조: 서론 → 본론 (3-5 섹션) → 결론
- 코드 예제 포함 (기술 글)
- 이미지/다이어그램 권장

### LinkedIn 포스트
- 길이: 150-300 단어
- 훅으로 시작 (질문 또는 통계)
- 핵심 인사이트 3개
- CTA로 마무리

### Twitter/X 스레드
- 메인 트윗: 핵심 메시지
- 스레드: 5-10개 트윗
- 각 트윗 독립적으로 이해 가능

## 학습된 선호도

(사용자 피드백에 따라 업데이트됨)
```

---

## Step 2: 스킬 정의

### skills/blog-writing/SKILL.md

```markdown
---
name: blog-writing
description: 블로그 포스트 작성 워크플로우
---

# Blog Writing Skill

## When to Use
- 사용자가 블로그 포스트 작성을 요청할 때
- 기술 튜토리얼이나 가이드 작성 시

## Workflow

### 1. 주제 분석
- 타겟 독자 파악
- 핵심 메시지 정의
- 키워드 리서치 (researcher 서브에이전트 활용)

### 2. 개요 작성
```
## [제목]

### 서론
- 독자의 문제/관심사 언급
- 글의 목적 제시

### 본론
#### 섹션 1: [주제]
#### 섹션 2: [주제]
#### 섹션 3: [주제]

### 결론
- 핵심 요약
- 다음 단계 제안
```

### 3. 초안 작성
- 섹션별로 작성
- 예제 코드/스크린샷 포함
- 내부 링크 추가

### 4. 검토 및 수정
- 가독성 확인
- SEO 최적화
- 문법/맞춤법 검사

## Output Format

```markdown
---
title: "제목"
slug: "url-slug"
description: "메타 설명"
date: YYYY-MM-DD
tags: [tag1, tag2]
---

# 제목

[본문]
```
```

### skills/social-media/SKILL.md

```markdown
---
name: social-media
description: 소셜 미디어 콘텐츠 작성 워크플로우
---

# Social Media Skill

## When to Use
- LinkedIn 포스트 작성 시
- Twitter/X 스레드 작성 시
- 소셜 미디어 캠페인 기획 시

## Platform Guidelines

### LinkedIn
- 전문적이면서 인간적인 톤
- 경험과 인사이트 공유
- 해시태그 3-5개

### Twitter/X
- 간결하고 임팩트 있게
- 이모지 적절히 사용
- 해시태그 1-2개

## Workflow

### 1. 핵심 메시지 정의
- 한 문장으로 요약
- 독자에게 주는 가치

### 2. 훅 작성
- 질문형: "혹시 ~한 경험 있으신가요?"
- 통계형: "90%의 개발자가 ~"
- 도전형: "이것 모르면 ~"

### 3. 본문 작성
- 핵심 포인트 3개
- 구체적 예시
- 개인 경험 (있다면)

### 4. CTA 추가
- 댓글 유도
- 공유 요청
- 링크 (있다면)
```

---

## Step 3: 서브에이전트 정의

### subagents.yaml

```yaml
researcher:
  description: "콘텐츠 주제에 대한 리서치를 수행하는 에이전트"
  system_prompt: |
    당신은 콘텐츠 작성을 위한 리서치를 수행합니다.

    ## 역할
    - 주제에 대한 최신 정보 수집
    - 트렌드 및 통계 파악
    - 경쟁 콘텐츠 분석

    ## 출력 형식
    - 핵심 발견사항 요약
    - 인용 가능한 통계
    - 참고 링크 목록
  tools:
    - web_search
  model: "anthropic:claude-sonnet-4-5-20250929"
```

---

## Step 4: 메인 에이전트 정의

### content_writer.py

```python
"""Content Builder Agent - 파일 기반 설정으로 콘텐츠 생성"""

import os
from pathlib import Path

import yaml
from langchain_core.tools import tool

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend


EXAMPLE_DIR = Path(__file__).parent


# === 도구 정의 ===

@tool
def web_search(
    query: str,
    max_results: int = 5,
) -> dict:
    """웹에서 정보를 검색합니다.

    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수
    """
    try:
        from tavily import TavilyClient

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return {"error": "TAVILY_API_KEY not set"}

        client = TavilyClient(api_key=api_key)
        return client.search(query, max_results=max_results)
    except Exception as e:
        return {"error": f"Search failed: {e}"}


# === 서브에이전트 로딩 ===

def load_subagents(config_path: Path) -> list:
    """YAML 파일에서 서브에이전트 정의를 로드합니다.

    Note:
        이것은 이 예제를 위한 커스텀 유틸리티입니다.
        memory와 skills와 달리, deepagents는 서브에이전트를
        파일에서 네이티브로 로드하지 않습니다.
    """
    available_tools = {
        "web_search": web_search,
    }

    with open(config_path) as f:
        config = yaml.safe_load(f)

    subagents = []
    for name, spec in config.items():
        subagent = {
            "name": name,
            "description": spec["description"],
            "system_prompt": spec["system_prompt"],
        }
        if "model" in spec:
            subagent["model"] = spec["model"]
        if "tools" in spec:
            subagent["tools"] = [available_tools[t] for t in spec["tools"]]
        subagents.append(subagent)

    return subagents


# === 에이전트 생성 ===

def create_content_writer():
    """파일 시스템 설정으로 콘텐츠 작성 에이전트를 생성합니다."""
    return create_deep_agent(
        memory=["./AGENTS.md"],           # 브랜드 보이스 로드
        skills=["./skills/"],             # 작성 워크플로우 로드
        tools=[web_search],               # 검색 도구
        subagents=load_subagents(EXAMPLE_DIR / "subagents.yaml"),
        backend=FilesystemBackend(root_dir=EXAMPLE_DIR),
    )


# === 실행 함수 ===

def create_content(prompt: str) -> str:
    """콘텐츠를 생성합니다.

    Args:
        prompt: 콘텐츠 요청 (예: "AI 에이전트에 대한 블로그 포스트 작성")

    Returns:
        생성된 콘텐츠
    """
    agent = create_content_writer()
    result = agent.invoke({
        "messages": [{"role": "user", "content": prompt}]
    })
    return result["messages"][-1].content


# === CLI ===

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = "AI 에이전트가 소프트웨어 개발을 어떻게 변화시키고 있는지에 대한 블로그 포스트를 작성해주세요"

    print(f"📝 요청: {prompt}\n")
    print("-" * 50)
    print(create_content(prompt))
```

**코드 상세 설명**:

```python
memory=["./AGENTS.md"],
```
- 브랜드 보이스와 스타일 가이드 로드
- 에이전트가 일관된 톤 유지

```python
skills=["./skills/"],
```
- `blog-writing`, `social-media` 스킬 자동 발견
- 에이전트가 적절한 워크플로우 선택

```python
subagents=load_subagents(EXAMPLE_DIR / "subagents.yaml"),
```
- YAML에서 서브에이전트 정의 로드
- 코드 변경 없이 설정 수정 가능

---

## Step 5: 실행

```bash
# 블로그 포스트
python content_writer.py "Python의 async/await에 대한 기술 블로그 포스트 작성"

# LinkedIn 포스트
python content_writer.py "AI 에이전트 트렌드에 대한 LinkedIn 포스트 작성"

# Twitter 스레드
python content_writer.py "프롬프트 엔지니어링 팁 Twitter 스레드 작성"
```

### 예상 출력 (블로그 포스트)

```
📝 요청: AI 에이전트에 대한 블로그 포스트 작성

--------------------------------------------------

먼저 AI 에이전트 트렌드에 대해 리서치하겠습니다.

[task 호출: researcher 서브에이전트]

리서치 결과를 바탕으로 블로그 포스트를 작성합니다.

---
title: "AI 에이전트: 소프트웨어 개발의 새로운 패러다임"
slug: "ai-agents-software-development"
description: "AI 에이전트가 개발자의 작업 방식을 어떻게 변화시키고 있는지 알아봅니다"
date: 2024-01-15
tags: [AI, 에이전트, 소프트웨어개발, LLM]
---

# AI 에이전트: 소프트웨어 개발의 새로운 패러다임

프로그래밍의 미래는 코드를 직접 작성하는 것이 아니라,
AI 에이전트와 협업하는 것일지도 모릅니다...

## AI 에이전트란?

AI 에이전트는 단순한 챗봇을 넘어...

[계속]
```

---

## 고급 기능

### 이미지 생성 통합

```python
@tool
def generate_cover(prompt: str, slug: str) -> str:
    """블로그 포스트용 커버 이미지를 생성합니다."""
    from google import genai

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt],
    )

    for part in response.parts:
        if part.inline_data is not None:
            image = part.as_image()
            output_path = EXAMPLE_DIR / "blogs" / slug / "hero.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(str(output_path))
            return f"Image saved to {output_path}"

    return "No image generated"
```

### 스트리밍 출력

```python
import asyncio

async def stream_content(prompt: str):
    """실시간으로 콘텐츠 생성 진행 상황을 출력합니다."""
    agent = create_content_writer()

    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": prompt}]},
        stream_mode="values",
    ):
        if "messages" in chunk:
            latest = chunk["messages"][-1]

            # 도구 호출 표시
            if hasattr(latest, "tool_calls") and latest.tool_calls:
                for tc in latest.tool_calls:
                    name = tc.get("name")
                    if name == "task":
                        print(f"🔄 리서치 중...")
                    elif name == "write_file":
                        print(f"📝 파일 저장 중...")

            # 텍스트 출력
            if hasattr(latest, "content") and latest.content:
                print(latest.content)

asyncio.run(stream_content("AI 에이전트 블로그 포스트 작성"))
```

---

## 문제 해결

### 스킬을 찾지 못함

```python
# 경로 확인
backend = FilesystemBackend(root_dir=EXAMPLE_DIR)
# skills 파라미터는 root_dir 기준 상대 경로
skills=["./skills/"]  # EXAMPLE_DIR/skills/
```

### 브랜드 보이스가 적용되지 않음

AGENTS.md 파일이 올바른 위치에 있는지 확인:
```python
memory=["./AGENTS.md"]  # EXAMPLE_DIR/AGENTS.md
```

---

## 다음 단계

- [메모리 시스템 상세](../02-core-concepts/memory-system.md)
- [스킬 시스템 상세](../02-core-concepts/skills-system.md)
- [컨텍스트 관리 패턴](../04-patterns/context-management.md)
