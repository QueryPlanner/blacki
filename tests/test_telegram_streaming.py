from blacki.telegram.streaming import _get_open_entities, split_long_message


def test_get_open_entities() -> None:
    assert _get_open_entities("hello *world") == ["*"]
    assert _get_open_entities("hello *world*") == []
    assert _get_open_entities("hello *world _and_") == ["*"]
    assert _get_open_entities("hello *world _and") == ["*", "_"]
    assert _get_open_entities("hello *world `and") == ["*", "`"]
    assert _get_open_entities("hello *world ```and") == ["*", "```"]
    assert _get_open_entities("hello *world ```and*") == [
        "*",
        "```",
    ]  # * inside code block is ignored
    assert _get_open_entities("hello \\*world") == []  # escaped
    assert _get_open_entities("hello ||spoiler") == ["||"]
    assert _get_open_entities("hello __underline") == ["__"]


def test_split_long_message_basic() -> None:
    # Basic split without entities
    text = "hello world that is very long"
    chunks = split_long_message(text, limit=15)
    assert chunks == ["hello world", "that is very", "long"]


def test_split_long_message_with_bold() -> None:
    text = "hello *bold text that is very long*"
    chunks = split_long_message(text, limit=20)
    assert chunks == ["hello *bold text*", "*that is very long*"]


def test_split_long_message_with_multiple_entities() -> None:
    text = "hello *bold _and italic_ text*"
    chunks = split_long_message(text, limit=20)
    assert chunks == ["hello *bold _and_*", "*_italic_ text*"]


def test_split_long_message_with_code_block() -> None:
    text = "hello ```python code block that is very long```"
    chunks = split_long_message(text, limit=30)
    assert chunks == ["hello ```python code block```", "```that is very long```"]


def test_get_open_entities_coverage() -> None:
    # Cover popping ```
    assert _get_open_entities("```code```") == []
    # Cover ` inside ```
    assert _get_open_entities("```code `inline` ```") == []
    # Cover ``` inside `
    assert _get_open_entities("`inline ``` code`") == []
    # Cover popping `
    assert _get_open_entities("`inline`") == []
    # Cover popping __ and ||
    assert _get_open_entities("__underline__") == []
    assert _get_open_entities("||spoiler||") == []
    assert _get_open_entities("~strikethrough~") == []


def test_split_long_message_hard_cut() -> None:
    # Text with no spaces, forcing a hard cut
    # Limit is 20, first char is *, so it opens bold.
    text = "*bold" + "A" * 40
    chunks = split_long_message(text, limit=20)
    assert chunks[0] == "*boldAAAAAAAAAAAAAA*"
    assert chunks[1] == "*AAAAAAAAAAAAAAAAAA*"
    assert chunks[2] == "*AAAAAAAA"
    assert len(chunks[0]) == 20
    assert len(chunks[1]) == 20
    assert len(chunks[2]) == 9
    assert chunks[0].startswith("*") and chunks[0].endswith("*")
    assert chunks[1].startswith("*") and chunks[1].endswith("*")


def test_split_long_message_boundary_recalculation() -> None:
    # Limit is 20.
    # text length is > 20. Space at index 20 to trigger split_index=20.
    # Space at index 9 to trigger fallback new_split_index=9 < max_allowed (19).
    text = "*boldA    BBBBBBBBBB " + "C" * 10
    # indices:
    # 012345678901234567890
    # *boldA    BBBBBBBBBB
    # 0..5 = *boldA
    # 6..9 = spaces
    # 10..19 = B's
    # 20 = space
    chunks = split_long_message(text, limit=20)
    # 1st try: split_index = 20. chunk = "*boldA    BBBBBBBBBB" (len 20)
    # closing_tags = "*"
    # len + closing = 21 > 20.
    # max_allowed = 19.
    # new_split_index = rfind(" ", 0, 20) -> index 9 (the last space in "    ")
    # new_split_index (9) < split_index (20) and != max_allowed (19). Hits else branch!
    # split_index = 9.
    # chunk = text[:9].rstrip() -> "*boldA"
    assert chunks[0] == "*boldA*"
    # remaining = "*" + "BBBBBBBBBB CCCCCCCCCC"
    # "*BBBBBBBBBB CCCCCCCCCC" (len 22)
    # split_index = rfind(" ", 0, 21) -> index 11 (between B and C)
    # chunk = "*BBBBBBBBBB"
    assert chunks[1] == "*BBBBBBBBBB*"
    assert chunks[2] == "*CCCCCCCCCC"


def test_split_long_message_infinite_loop_prevention() -> None:
    # If limit is so small that closing tags exceed the limit, it could infinite loop.
    # We set a limit of 2, and have 3 open entities.
    text = "*_~A"
    chunks = split_long_message(text, limit=2)
    # remaining = "*_~A", limit=2. chunk="*_". entities=["*", "_"]. closing="_*".
    # len(chunk) + len(closing) = 2 + 2 = 4 > 2.
    # It hits `len(closing_tags) >= limit` which breaks the loop.
    # Then chunk="*_", entities=[], closing="".
    assert chunks[0] == "*_"
    assert chunks[1] == "~A"


def test_split_long_message_keeps_escape_pairs_together() -> None:
    """A limit boundary cannot strand a Markdown escape backslash."""
    chunks = split_long_message(r"aaaa\*bbbb", limit=5)

    assert chunks == ["aaaa", r"\*bbb", "b"]
    assert all(not chunk.endswith("\\") for chunk in chunks)
    assert all(_get_open_entities(chunk) == [] for chunk in chunks)
