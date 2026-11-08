from __future__ import annotations

from blacki.miniapp.catalog import (
    ACTIONS,
    AGENT_HOME_PAGE,
    BUILTIN_PAGES,
    CALLOUT_TONES,
    COMPONENTS,
    FUNCTIONS,
    REDUCERS,
    TARGET_KINDS,
    TONES,
    ComponentSpec,
    Param,
    manifest,
    reference_text,
)


def test_catalog_constants() -> None:
    assert "default" in TONES
    assert "info" in CALLOUT_TONES
    assert "protein_g" in TARGET_KINDS
    assert "today" in BUILTIN_PAGES
    assert AGENT_HOME_PAGE == "foryou"
    assert "Sum" in REDUCERS
    assert "Page" in COMPONENTS
    assert "AskBlacki" in ACTIONS
    assert "Round" in FUNCTIONS


def test_param_doc_branches() -> None:
    # Param with options, min/max, allowed, default, required=True vs False
    p1 = Param("p1", "enum", required=True, options=("a", "b"))
    p2 = Param("p2", "number", min=0.0, max=100.0, default=10)
    p3 = Param("p3", "children", allowed=("Card", "Text"))
    p4 = Param("p4", "text", min=0.0)  # only min, max is None
    spec = ComponentSpec(
        "Custom", "Test", "Test doc", (p1, p2, p3, p4), example="Custom()"
    )

    assert spec.param("p1") == p1
    assert spec.param("nonexistent") is None


def test_manifest_output() -> None:
    m = manifest()
    assert isinstance(m, dict)
    assert "components" in m
    assert "actions" in m
    assert "functions" in m
    assert "tones" in m
    assert "Page" in m["components"]
    assert "AskBlacki" in m["actions"]
    assert "Round" in m["functions"]
    assert list(TONES) == m["tones"]


def test_reference_text_output() -> None:
    ref = reference_text()
    assert "COMPONENTS" in ref
    assert "FUNCTIONS" in ref
    assert "ACTIONS" in ref
    assert "TONES" in ref
    assert "Page [Layout]" in ref
    assert "@AskBlacki" in ref
