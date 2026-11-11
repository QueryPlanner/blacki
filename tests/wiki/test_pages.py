"""Pure wiki rules: paths, edits, secret detection, headings, and search."""

import pytest

from blacki.wiki.pages import (
    PAGE_CHAR_LIMIT,
    SEARCH_LINE_LIMIT,
    EditError,
    Page,
    apply_edit,
    contains_secret,
    heading_of,
    normalize_path,
    search_pages,
)


def _page(path: str, markdown: str) -> Page:
    return Page(path, markdown, 1, "2026-11-09T00:00:00+00:00")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Alice", "alice.md"),
        ("  People/Alice.MD  ", "people/alice.md"),
        ("people/alice_smith.md", "people/alice_smith.md"),
        ("projects/blacki-memory/notes", "projects/blacki-memory/notes.md"),
        ("café/notas", "café/notas.md"),
        ("x" * 157, "x" * 157 + ".md"),
    ],
)
def test_normalize_path_returns_canonical_relative_paths(
    raw: str, expected: str
) -> None:
    assert normalize_path(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        ".md",
        "../escape",
        "/absolute.md",
        "a/../b.md",
        "a b.md",
        "a..md",
        "notes.md.md",
        "folder/",
        "x" * 158,
    ],
)
def test_normalize_path_rejects_unsafe_or_overlong_paths(raw: str) -> None:
    assert normalize_path(raw) is None


def test_apply_edit_appends_on_a_new_line() -> None:
    assert apply_edit("- Likes tea", "- Likes coffee", "") == (
        "- Likes tea\n- Likes coffee"
    )
    assert apply_edit("- Likes tea\n", "- Likes coffee", "") == (
        "- Likes tea\n- Likes coffee"
    )


def test_apply_edit_creates_text_from_an_empty_page() -> None:
    assert apply_edit("", "# Ana\n", "") == "# Ana\n"


def test_apply_edit_inserts_under_a_heading() -> None:
    current = "## Food\n- Likes tea\n"
    assert apply_edit(current, "## Food\n- Vegetarian\n", "## Food\n") == (
        "## Food\n- Vegetarian\n- Likes tea\n"
    )


def test_apply_edit_replaces_exactly_one_occurrence() -> None:
    assert apply_edit("- Lives in Pune", "Mumbai", "Pune") == "- Lives in Mumbai"


def test_apply_edit_deletes_matched_text_when_new_text_is_empty() -> None:
    assert apply_edit("keep\nremove me\n", "", "remove me\n") == "keep\n"


@pytest.mark.parametrize(
    ("current", "new_text", "old_text", "status"),
    [
        ("- tea", "- coffee", "- milk", "no_match"),
        ("- tea\n- tea", "- coffee", "- tea", "ambiguous"),
        ("- tea", "   ", "", "invalid_request"),
        ("- tea", "", "", "invalid_request"),
    ],
)
def test_apply_edit_rejects_with_an_actionable_status(
    current: str, new_text: str, old_text: str, status: str
) -> None:
    with pytest.raises(EditError) as error:
        apply_edit(current, new_text, old_text)
    assert error.value.status == status
    assert error.value.hint


def test_apply_edit_enforces_the_page_limit_on_the_resulting_page() -> None:
    current = "a" * (PAGE_CHAR_LIMIT - 3)
    # The append adds one separator newline, so two characters reach the limit.
    assert len(apply_edit(current, "bb", "")) == PAGE_CHAR_LIMIT
    with pytest.raises(EditError) as error:
        apply_edit(current, "bbb", "")
    assert error.value.status == "too_large"


@pytest.mark.parametrize(
    "text",
    [
        "My password is hunter2",
        "The passwords are in the vault",
        "API key: abc",
        "api_key=abc",
        "the access token expires",
        "secret key rotation",
        "passphrase is ordinary words",
        "don't remember this",
        "sk-abcdefghijklmnop",
        "ghp_abcdefghijkl",
        "xoxb-abcdefghijklmn",
    ],
)
def test_contains_secret_flags_credentials(text: str) -> None:
    assert contains_secret(text)


@pytest.mark.parametrize(
    "text",
    ["Prefers short replies.", "Passport renewal is due", "Likes ask-a-question"],
)
def test_contains_secret_allows_ordinary_text(text: str) -> None:
    assert not contains_secret(text)


def test_heading_of_returns_the_first_heading_line() -> None:
    assert heading_of("intro\n## Food\n# Later") == "## Food"
    assert heading_of("no heading here") == ""


def test_search_requires_every_term_in_the_path_or_body() -> None:
    pages = [
        _page("food.md", "Likes tea and cake"),
        _page("work.md", "Likes tea"),
    ]
    assert [hit.path for hit in search_pages(pages, "tea cake")] == ["food.md"]


def test_search_ranks_by_total_hits_then_path() -> None:
    pages = [
        _page("b.md", "tea once"),
        _page("a.md", "tea tea tea"),
        _page("c.md", "tea tea"),
    ]
    assert [hit.path for hit in search_pages(pages, "TEA")] == [
        "a.md",
        "c.md",
        "b.md",
    ]


def test_search_matches_terms_in_the_path_with_no_matching_lines() -> None:
    (hit,) = search_pages([_page("projects/Blacki.md", "Notes")], "blacki")
    assert hit.path == "projects/Blacki.md"
    assert hit.lines == ()


def test_search_returns_matching_lines_with_one_line_of_context() -> None:
    markdown = "one\ntwo\nthree target\nfour\nfive\nsix target\nseven"
    (hit,) = search_pages([_page("notes.md", markdown)], "target")
    assert [(line.number, line.text, line.matched) for line in hit.lines] == [
        (2, "two", False),
        (3, "three target", True),
        (4, "four", False),
        (5, "five", False),
        (6, "six target", True),
        (7, "seven", False),
    ]


def test_search_caps_the_lines_returned_for_one_page() -> None:
    markdown = "\n".join(f"target {number}" for number in range(40))
    (hit,) = search_pages([_page("notes.md", markdown)], "target")
    assert len(hit.lines) == SEARCH_LINE_LIMIT
