"""MarkdownV2 formatting utilities for Telegram messages.

Provides escaping and formatting functions for Telegram's MarkdownV2 syntax,
which requires special handling for reserved characters.
"""

import re

MARKDOWN_SPECIAL_CHARS = frozenset("_*[]()~>#+-=|{}.!\\")

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
BULLET_PATTERN = re.compile(r"^(\s*)[*\-+]\s+", re.MULTILINE)


def escape_markdown(text: str) -> str:
    """Escape special Markdown characters for Telegram MarkdownV2.

    Does NOT escape inside code blocks or inline code - those are preserved.
    """
    result: list[str] = []
    in_code_block = False
    in_inline_code = False
    i = 0

    while i < len(text):
        char = text[i]

        if i + 2 <= len(text) and text[i : i + 3] == "```":
            in_code_block = not in_code_block
            result.append("```")
            i += 3
            continue

        if char == "`" and not in_code_block:
            in_inline_code = not in_inline_code
            result.append(char)
            i += 1
            continue

        if not in_code_block and not in_inline_code:
            if char in MARKDOWN_SPECIAL_CHARS:
                result.append("\\")
                result.append(char)
            else:
                result.append(char)
        else:
            result.append(char)

        i += 1

    return "".join(result)


def format_for_telegram(text: str) -> str:
    """Format text for Telegram MarkdownV2, converting markdown to native format.

    Converts:
    - **bold** to *bold* (Telegram bold)
    - # Heading to *Heading* (bold, no heading in Telegram)
    - * item, - item to • item (bullet character)
    - Escapes remaining special characters
    """
    text = _convert_headings_to_bold(text)
    text = _convert_bullets(text)
    text = _convert_bold(text)
    text = _escape_remaining(text)
    return text


def _convert_headings_to_bold(text: str) -> str:
    """Convert markdown headings to bold text."""

    def replace_heading(match: re.Match[str]) -> str:
        heading_text = match.group(2)
        return f"**{heading_text}**"

    return HEADING_PATTERN.sub(replace_heading, text)


def _convert_bullets(text: str) -> str:
    """Convert markdown bullets to Telegram bullet character."""

    def replace_bullet(match: re.Match[str]) -> str:
        indent = match.group(1)
        return f"{indent}• "

    return BULLET_PATTERN.sub(replace_bullet, text)


def _convert_bold(text: str) -> str:
    """Convert **bold** to *bold* for Telegram, handling nested code.

    Unclosed ** markers are escaped as \\*\\*.
    """
    result: list[str] = []
    i = 0
    in_code_block = False
    in_inline_code = False

    while i < len(text):
        if i + 2 <= len(text) and text[i : i + 3] == "```":
            in_code_block = not in_code_block
            result.append("```")
            i += 3
            continue

        if text[i] == "`" and not in_code_block:
            in_inline_code = not in_inline_code
            result.append("`")
            i += 1
            continue

        if (
            not in_code_block
            and not in_inline_code
            and i + 1 < len(text)
            and text[i : i + 2] == "**"
        ):
            j = i + 2
            inner_in_code_block = False
            inner_in_inline_code = False
            while j + 1 < len(text):
                if j + 2 <= len(text) and text[j : j + 3] == "```":
                    inner_in_code_block = not inner_in_code_block
                    j += 3
                    continue
                if text[j] == "`" and not inner_in_code_block:
                    inner_in_inline_code = not inner_in_inline_code
                    j += 1
                    continue
                if (
                    not inner_in_code_block
                    and not inner_in_inline_code
                    and j + 1 < len(text)
                    and text[j : j + 2] == "**"
                ):
                    break
                j += 1

            if j + 1 < len(text) and text[j : j + 2] == "**":
                bold_content = text[i + 2 : j]
                result.append(f"*{bold_content}*")
                i = j + 2
                continue
            result.append("\\*\\*")
            i += 2
            continue

        result.append(text[i])
        i += 1

    return "".join(result)


def _escape_remaining(text: str) -> str:
    """Escape remaining special characters, preserving code blocks and escapes."""
    result: list[str] = []
    i = 0
    in_code_block = False
    in_inline_code = False
    in_bold = False

    while i < len(text):
        if i + 2 <= len(text) and text[i : i + 3] == "```":
            in_code_block = not in_code_block
            result.append("```")
            i += 3
            continue

        if text[i] == "`" and not in_code_block:
            in_inline_code = not in_inline_code
            result.append("`")
            i += 1
            continue

        if text[i] == "*" and not in_code_block and not in_inline_code:
            in_bold = not in_bold
            result.append("*")
            i += 1
            continue

        if (
            not in_code_block
            and not in_inline_code
            and text[i] in MARKDOWN_SPECIAL_CHARS
        ):
            if text[i] == "\\" and i + 1 < len(text) and text[i + 1] == "*":
                result.append("\\")
                result.append("*")
                i += 2
                continue
            result.append("\\")
        result.append(text[i])
        i += 1

    return "".join(result)
