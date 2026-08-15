"""MarkdownV2 formatting utilities for Telegram messages.

Provides escaping and formatting functions for Telegram's MarkdownV2 syntax,
which requires special handling for reserved characters.
"""

import re
import unicodedata

MARKDOWN_SPECIAL_CHARS = frozenset("_*[]()~`>#+-=|{}.!\\")

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
BULLET_PATTERN = re.compile(r"^(\s*)[*\-+]\s+", re.MULTILINE)
TABLE_SEPARATOR_CELL_PATTERN = re.compile(r"^:?-+:?$")


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
    - Markdown tables to aligned code blocks (Telegram has no table entity)
    - Escapes remaining special characters
    """
    original_text = text
    text = _convert_markdown_tables(text)
    text = _convert_headings_to_bold(text)
    text = _convert_bullets(text)
    text = _convert_bold(text)
    text = _escape_remaining(text)
    if get_open_markdown_entities(text):
        return escape_markdown_plain(original_text)
    return text


def escape_markdown_plain(text: str) -> str:
    """Escape every MarkdownV2 control character without preserving entities."""
    return "".join(
        f"\\{char}" if char in MARKDOWN_SPECIAL_CHARS else char for char in text
    )


def get_open_markdown_entities(text: str) -> list[str]:
    """Return unclosed MarkdownV2 entities while respecting escaped markers."""
    open_entities: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue

        if text[i : i + 3] == "```":
            if "`" in open_entities:
                i += 3
                continue
            if open_entities and open_entities[-1] == "```":
                open_entities.pop()
            else:
                open_entities.append("```")
            i += 3
            continue

        if text[i] == "`":
            if "```" in open_entities:
                i += 1
                continue
            if open_entities and open_entities[-1] == "`":
                open_entities.pop()
            else:
                open_entities.append("`")
            i += 1
            continue

        if "```" in open_entities or "`" in open_entities:
            i += 1
            continue

        if text[i : i + 2] in ("__", "||"):
            marker = text[i : i + 2]
            if open_entities and open_entities[-1] == marker:
                open_entities.pop()
            else:
                open_entities.append(marker)
            i += 2
            continue

        if text[i] in ("*", "_", "~"):
            marker = text[i]
            if open_entities and open_entities[-1] == marker:
                open_entities.pop()
            else:
                open_entities.append(marker)
            i += 1
            continue

        i += 1

    return open_entities


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


def _convert_markdown_tables(text: str) -> str:
    """Convert GitHub-style Markdown tables to Telegram code blocks."""
    if "|" not in text:
        return text

    lines = text.splitlines()
    newline = "\r\n" if "\r\n" in text else "\n"
    has_trailing_newline = text.endswith(("\n", "\r"))
    converted: list[str] = []
    in_code_block = False
    index = 0

    while index < len(lines):
        line = lines[index]
        if line.lstrip().startswith("```"):
            converted.append(line)
            in_code_block = not in_code_block
            index += 1
            continue

        if not in_code_block and index + 1 < len(lines):
            header = _split_table_row(line)
            separator = _split_table_row(lines[index + 1])
            if (
                header
                and separator
                and len(header) == len(separator)
                and all(
                    TABLE_SEPARATOR_CELL_PATTERN.fullmatch(cell.strip())
                    for cell in separator
                )
            ):
                rows = [header]
                index += 2
                while index < len(lines):
                    row = _split_table_row(lines[index])
                    if row is None:
                        break
                    rows.append(row)
                    index += 1

                converted.extend(["```", *_render_table(rows), "```"])
                continue

        converted.append(line)
        index += 1

    result = newline.join(converted)
    if has_trailing_newline:
        result += newline
    return result


def _split_table_row(line: str) -> list[str] | None:
    """Split a Markdown table row without treating escaped/code pipes as delimiters."""
    cells: list[str] = []
    current: list[str] = []
    code_delimiter_length: int | None = None
    has_pipe = False
    index = 0

    while index < len(line):
        character = line[index]
        if character == "`":
            run_start = index
            while index < len(line) and line[index] == "`":
                index += 1
            run_length = index - run_start
            current.extend("`" * run_length)
            if code_delimiter_length is None:
                code_delimiter_length = run_length
            elif run_length == code_delimiter_length:
                code_delimiter_length = None
            continue

        if character == "|" and code_delimiter_length is None:
            backslashes = 0
            preceding = index - 1
            while preceding >= 0 and line[preceding] == "\\":
                backslashes += 1
                preceding -= 1
            if backslashes % 2 == 0:
                cells.append("".join(current).strip())
                current = []
                has_pipe = True
                index += 1
                continue

        current.append(character)
        index += 1

    if not has_pipe:
        return None

    cells.append("".join(current).strip())
    if line.lstrip().startswith("|") and cells:
        cells.pop(0)
    if _has_unescaped_trailing_pipe(line) and cells:
        cells.pop()
    return cells or None


def _has_unescaped_trailing_pipe(line: str) -> bool:
    """Return whether the row ends with an unescaped table delimiter."""
    stripped = line.rstrip()
    if not stripped.endswith("|"):
        return False

    backslashes = 0
    preceding = len(stripped) - 2
    while preceding >= 0 and stripped[preceding] == "\\":
        backslashes += 1
        preceding -= 1
    return backslashes % 2 == 0


def _render_table(rows: list[list[str]]) -> list[str]:
    """Render table rows as padded text suitable for a Telegram code block."""
    column_count = max(len(row) for row in rows)
    normalized_rows = [
        [cell.replace(r"\|", "|") for cell in row] + [""] * (column_count - len(row))
        for row in rows
    ]
    widths = [
        max(1, max(_display_width(row[column]) for row in normalized_rows))
        for column in range(column_count)
    ]

    def render_row(row: list[str]) -> str:
        return " | ".join(
            _pad_cell(cell, width) for cell, width in zip(row, widths, strict=True)
        ).rstrip()

    rendered = [render_row(normalized_rows[0])]
    rendered.append("-+-".join("-" * width for width in widths))
    rendered.extend(render_row(row) for row in normalized_rows[1:])
    return rendered


def _display_width(text: str) -> int:
    """Return the approximate monospace display width of Unicode text."""
    width = 0
    for character in text:
        category = unicodedata.category(character)
        if unicodedata.combining(character) or category in {"Cf", "Mn", "Me"}:
            continue
        width += 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
    return width


def _pad_cell(cell: str, width: int) -> str:
    """Pad a table cell to a target display width."""
    return cell + " " * max(0, width - _display_width(cell))


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

        if ((in_code_block or in_inline_code) and text[i] == "\\") or (
            in_code_block and text[i] == "`"
        ):
            result.append("\\")

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
