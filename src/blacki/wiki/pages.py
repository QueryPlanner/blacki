"""Pure rules for private wiki pages: paths, edits, secrets, and search."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

PAGE_CHAR_LIMIT = 15_000
MAX_PATH_LENGTH = 160
MAX_QUERY_LENGTH = 180
SEARCH_RESULT_LIMIT = 10
SEARCH_LINE_LIMIT = 20

_PATH_PATTERN = re.compile(r"(?:[\w-]+/)*[\w-]+\.md")
_SECRET_PATTERN = re.compile(
    r"\b(?:passwords?|passphrase|api[ _-]?key|access token|secret key|"
    r"don't remember|do not remember)\b"
    r"|\b(?:sk-|ghp_|gho_|xox[baprs]-)[\w-]{12,}",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Page:
    """One owner-scoped Markdown page as stored."""

    path: str
    markdown: str
    revision: int
    updated_at: str


class EditError(Exception):
    """An edit cannot be applied. The status and hint tell the agent what to do."""

    def __init__(self, status: str, hint: str) -> None:
        super().__init__(status)
        self.status = status
        self.hint = hint


@dataclass(frozen=True)
class PageLine:
    number: int
    text: str
    matched: bool


@dataclass(frozen=True)
class SearchHit:
    path: str
    score: int
    lines: tuple[PageLine, ...]


def normalize_path(raw: str) -> str | None:
    """Return the canonical page path, or None when the path is not allowed."""
    value = raw.strip().casefold()
    if not value.endswith(".md"):
        value += ".md"
    if len(value) > MAX_PATH_LENGTH or not _PATH_PATTERN.fullmatch(value):
        return None
    return value


def contains_secret(text: str) -> bool:
    """Return True when text looks like a password, API key, or token."""
    return _SECRET_PATTERN.search(text) is not None


def apply_edit(current: str, new_text: str, old_text: str) -> str:
    """Return the page text after one edit, or raise EditError.

    An empty old_text appends new_text on its own line. A non-empty old_text
    must occur exactly once and is replaced by new_text, so an empty new_text
    deletes it.
    """
    if old_text:
        matches = current.count(old_text)
        if matches == 0:
            raise EditError(
                "no_match",
                "old_text is not in the page. Call wiki_read on this path, copy "
                "the exact current text into old_text, and retry.",
            )
        if matches > 1:
            raise EditError(
                "ambiguous",
                f"old_text occurs {matches} times. Include more surrounding text "
                "so it occurs exactly once.",
            )
        markdown = current.replace(old_text, new_text, 1)
    else:
        if not new_text.strip():
            raise EditError(
                "invalid_request",
                "new_text is empty. Pass text to append, or pass old_text to "
                "replace or delete existing text.",
            )
        separator = "\n" if current and not current.endswith("\n") else ""
        markdown = current + separator + new_text
    if len(markdown) > PAGE_CHAR_LIMIT:
        raise EditError(
            "too_large",
            f"The page would be {len(markdown)} characters; the limit is "
            f"{PAGE_CHAR_LIMIT}. Move details to another page or remove outdated "
            "lines.",
        )
    return markdown


def heading_of(markdown: str) -> str:
    """Return the first Markdown heading line, or an empty string."""
    for line in markdown.splitlines():
        if line.lstrip().startswith("#"):
            return line.strip()
    return ""


def search_pages(pages: Iterable[Page], query: str) -> list[SearchHit]:
    """Return pages where every query term occurs in the path or body.

    Terms match case-insensitively. Hits rank by total occurrences, then path.
    """
    terms = [term.casefold() for term in query.split()]
    hits: list[SearchHit] = []
    for page in pages:
        path = page.path.casefold()
        body = page.markdown.casefold()
        if not all(term in path or term in body for term in terms):
            continue
        score = sum(path.count(term) + body.count(term) for term in terms)
        lines = page.markdown.splitlines()
        matched = [
            index
            for index, line in enumerate(lines)
            if any(term in line.casefold() for term in terms)
        ]
        hits.append(SearchHit(page.path, score, _lines_with_context(lines, matched)))
    hits.sort(key=lambda hit: (-hit.score, hit.path))
    return hits


def _lines_with_context(lines: list[str], matched: list[int]) -> tuple[PageLine, ...]:
    """Return matching lines plus one line of context each side, capped."""
    matched_set = set(matched)
    chosen: list[int] = []
    for index in matched:
        for near in range(max(0, index - 1), min(len(lines), index + 2)):
            if near not in chosen:
                chosen.append(near)
        if len(chosen) >= SEARCH_LINE_LIMIT:
            break
    return tuple(
        PageLine(index + 1, lines[index], index in matched_set)
        for index in sorted(chosen[:SEARCH_LINE_LIMIT])
    )
