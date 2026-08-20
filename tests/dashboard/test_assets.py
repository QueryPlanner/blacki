"""Focused contract checks for the bundled observability dashboard assets."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[2]
TEMPLATE = ROOT / "src/blacki/dashboard/templates/index.html"
SCRIPT = ROOT / "src/blacki/dashboard/static/dashboard.js"
STYLESHEET = ROOT / "src/blacki/dashboard/static/dashboard.css"


def test_dashboard_assets_are_bundled_without_runtime_cdn_dependencies() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")
    stylesheet = STYLESHEET.read_text(encoding="utf-8")

    combined = "\n".join((template, script, stylesheet))
    assert "cdn." not in combined.lower()
    assert "https://cdn." not in combined.lower()
    assert "https://" not in "\n".join((template, script)).lower()
    assert "<style" not in template.lower()
    assert "<script>" not in template.lower()
    assert '<script src="/dashboard/static/dashboard.js" defer></script>' in template
    assert stylesheet.strip()
    assert "@import" not in stylesheet


def test_dynamic_rendering_avoids_unsafe_html_apis() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    for forbidden in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
    ):
        assert forbidden not in script
    assert "textContent" in script
    assert "createElement" in script
    assert "replaceChildren" in script


def test_chat_metadata_events_render_as_safe_rows() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert 'type === "tool"' in script
    assert "metadataRow" in script
    assert "attachments" in script
    assert "attachmentValueIsPresent" in script
    assert 'el("div", `chat ' in script


def test_dashboard_renders_cumulative_and_monthly_cost_signals() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")

    for marker in (
        "metric-cost-month",
        "metric-cost-total",
        "metric-cost-average",
        "selected-user-cost",
        "selected-session-cost",
    ):
        assert marker in template
    for marker in (
        "formatMoney",
        "monthly_cost_usd",
        "monthly_estimated_cost_usd",
        "cumulative_cost_usd",
        "cumulative_estimated_cost_usd",
        "average_user_monthly_cost_usd",
        "cost_coverage",
    ):
        assert marker in script


def test_template_exposes_accessible_view_tabs_and_landmarks() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")

    for view in ("overview", "users", "traces", "logs"):
        assert f'data-view="{view}"' in template
        assert f'data-panel="{view}"' in template
    assert 'role="tablist"' in template
    assert 'role="tabpanel"' in template
    assert 'aria-selected="true"' in template
    assert 'aria-controls="overview-panel"' in template
    assert 'aria-label="Dashboard views"' in template
    assert 'id="refresh-button"' in template
    assert 'id="auto-refresh"' in template
    assert 'id="theme-toggle"' in template
    assert 'role="alert"' in template
    assert "/reset" in template


def test_uses_daisyui_v5_classes_in_source_and_compiled_css() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")
    stylesheet = STYLESHEET.read_text(encoding="utf-8")
    source = "\n".join((template, script))

    selected_components = (
        "navbar",
        "stats",
        "tabs",
        "tabs-box",
        "tab",
        "card",
        "table",
        "chat",
        "badge",
        "loading",
        "fieldset",
        "input",
        "select",
        "toggle",
        "timeline",
    )
    for component in selected_components:
        assert component in source
        assert f".{component}" in stylesheet
    forbidden_classes = ("tabs" + "-boxed", "input" + "-bordered")
    for forbidden in forbidden_classes:
        assert forbidden not in template


def test_reset_notice_uses_theme_safe_semantic_text_colors() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")

    assert "alert-info" not in template
    assert "border-info/30 bg-info/10 text-base-content" in template
    assert 'class="text-info text-lg"' in template
    assert 'class="font-semibold text-base-content"' in template
    assert "text-base-content/70" in template


def test_script_keeps_expected_same_origin_api_contract() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    expected_paths = (
        "/dashboard/api/overview?window=24h",
        "/dashboard/api/users?search=",
        "/dashboard/api/sessions?user_id=",
        "/dashboard/api/session?user_id=",
        "/dashboard/api/logs?level=",
        "/dashboard/api/traces?status=",
        "/dashboard/api/trace?trace_id=",
    )
    for path in expected_paths:
        assert path in script
    assert "fetch(url" in script
    assert 'cache: "no-store"' in script
    assert "localStorage" not in script


def test_frontend_dependencies_are_pinned() -> None:
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    dependencies = package["devDependencies"]

    assert dependencies["tailwindcss"] == "4.1.18"
    assert dependencies["@tailwindcss/cli"] == "4.1.18"
    assert dependencies["daisyui"] == "5.5.5"
    assert package["scripts"]["build"].startswith("tailwindcss -i ")
