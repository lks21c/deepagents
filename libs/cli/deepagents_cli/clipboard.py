"""
모듈명: clipboard.py
설명: deepagents-cli용 클립보드 유틸리티

주요 기능:
- copy_selection_to_clipboard(): 선택된 텍스트를 시스템 클립보드로 복사
- _copy_osc52(): OSC 52 이스케이프 시퀀스를 사용한 복사 (SSH/tmux 지원)

의존성:
- pyperclip: 크로스 플랫폼 클립보드 접근 (선택적)
- textual: TUI 앱 통합
"""

from __future__ import annotations

import base64
import logging
import os
import pathlib
from typing import TYPE_CHECKING

from deepagents_cli.config import get_glyphs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from textual.app import App

_PREVIEW_MAX_LENGTH = 40


def _copy_osc52(text: str) -> None:
    """Copy text using OSC 52 escape sequence (works over SSH/tmux)."""
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    osc52_seq = f"\033]52;c;{encoded}\a"
    if os.environ.get("TMUX"):
        osc52_seq = f"\033Ptmux;\033{osc52_seq}\033\\"

    with pathlib.Path("/dev/tty").open("w", encoding="utf-8") as tty:
        tty.write(osc52_seq)
        tty.flush()


def _shorten_preview(texts: list[str]) -> str:
    """Shorten text for notification preview.

    Returns:
        Shortened preview text suitable for notification display.
    """
    glyphs = get_glyphs()
    dense_text = glyphs.newline.join(texts).replace("\n", glyphs.newline)
    if len(dense_text) > _PREVIEW_MAX_LENGTH:
        return f"{dense_text[: _PREVIEW_MAX_LENGTH - 1]}{glyphs.ellipsis}"
    return dense_text


def copy_selection_to_clipboard(app: App) -> None:
    """Copy selected text from app widgets to clipboard.

    This queries all widgets for their text_selection and copies
    any selected text to the system clipboard.
    """
    selected_texts = []

    for widget in app.query("*"):
        if not hasattr(widget, "text_selection") or not widget.text_selection:
            continue

        selection = widget.text_selection

        try:
            result = widget.get_selection(selection)
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(
                "Failed to get selection from widget %s: %s",
                type(widget).__name__,
                e,
                exc_info=True,
            )
            continue

        if not result:
            continue

        selected_text, _ = result
        if selected_text.strip():
            selected_texts.append(selected_text)

    if not selected_texts:
        return

    combined_text = "\n".join(selected_texts)

    # Try multiple clipboard methods
    # Prefer pyperclip/app clipboard first (works reliably on local machines)
    # OSC 52 is last resort (for SSH/remote where native clipboard unavailable)
    copy_methods = [app.copy_to_clipboard]

    # Try pyperclip if available (preferred - uses pbcopy on macOS)
    try:
        import pyperclip

        copy_methods.insert(0, pyperclip.copy)
    except ImportError:
        pass

    # OSC 52 as fallback for remote/SSH sessions
    copy_methods.append(_copy_osc52)

    for copy_fn in copy_methods:
        try:
            copy_fn(combined_text)
            # Use markup=False to prevent copied text from being parsed as Rich markup
            app.notify(
                f'"{_shorten_preview(selected_texts)}" copied',
                severity="information",
                timeout=2,
                markup=False,
            )
        except (OSError, RuntimeError, TypeError) as e:
            logger.debug(
                "Clipboard copy method %s failed: %s",
                getattr(copy_fn, "__name__", repr(copy_fn)),
                e,
                exc_info=True,
            )
            continue
        else:
            return

    # If all methods fail, still notify but warn
    app.notify(
        "Failed to copy - no clipboard method available",
        severity="warning",
        timeout=3,
    )
