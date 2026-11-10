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


def test_dashboard_navigation_and_shared_theme_are_packaged() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")

    assert 'href="/dashboard/static/miniapp.css"' in template
    assert 'data-theme="blacki"' in template
    assert 'aria-label="Dashboard views"' in template
    for view in ("overview", "sessions", "traces", "logs"):
        assert f'href="#{view}"' in template
    for control in ("refresh", "auto-refresh", "theme-toggle"):
        assert f'id="{control}"' in template


def test_frontend_dependencies_are_pinned() -> None:
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    dependencies = package["devDependencies"]

    assert dependencies["tailwindcss"] == "4.1.18"
    assert dependencies["@tailwindcss/cli"] == "4.1.18"
    assert dependencies["daisyui"] == "5.5.5"
    assert package["scripts"]["build"].startswith("tailwindcss -i ")
