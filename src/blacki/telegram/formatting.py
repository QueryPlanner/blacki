"""MarkdownV2 formatting utilities for Telegram messages.

Provides escaping and formatting functions for Telegram's MarkdownV2 syntax,
which requires special handling for reserved characters.
"""

MARKDOWN_SPECIAL_CHARS = frozenset("_*[]()~>#+-=|{}.!\\")


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
    """Format text for Telegram MarkdownV2, preserving bold formatting.

    Handles **bold** markdown by:
    1. Identifying bold sections
    2. Escaping special chars in all content
    3. Converting ** to * for Telegram bold syntax
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

        if not in_code_block and not in_inline_code:
            if i + 1 < len(text) and text[i : i + 2] == "**":
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
                    escaped_content = _escape_text_only(bold_content)
                    result.append(f"*{escaped_content}*")
                    i = j + 2
                    continue

            if text[i] in MARKDOWN_SPECIAL_CHARS:
                result.append("\\")
            result.append(text[i])
        else:
            result.append(text[i])

        i += 1

    return "".join(result)


def _escape_text_only(text: str) -> str:
    """Escape special chars without code block handling (for internal use)."""
    result: list[str] = []
    in_code_block = False
    in_inline_code = False
    i = 0

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

        if not in_code_block and not in_inline_code:
            if text[i] in MARKDOWN_SPECIAL_CHARS:
                result.append("\\")
            result.append(text[i])
        else:
            result.append(text[i])

        i += 1

    return "".join(result)
