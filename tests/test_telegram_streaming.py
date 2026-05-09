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
    # We want a split at 20 that exceeds limit when closing tags are added,
    # but a smaller boundary (space) exists to fallback to.
    text = "*bold" + " " + "A" * 15
    chunks = split_long_message(text, limit=20)
    # len is 21. split_index = 20. chunk = "*bold AAAAAAAAAAAAAA"
    # len + closing = 20 + 1 = 21 > 20
    # max_allowed = 19. _find_chunk_boundary(19) finds space at index 5.
    # Hits else block: split_index = 5.
    # chunk = "*bold"
    assert chunks[0] == "*bold*"
    assert chunks[1] == "*AAAAAAAAAAAAAAA"
