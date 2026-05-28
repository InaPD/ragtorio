"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "wikitext" / "factorio"


@pytest.fixture
def wikitext() -> dict[str, str]:
    """Every committed Factorio fixture, keyed by filename stem."""
    return {p.stem: p.read_text(encoding="utf-8") for p in FIXTURE_DIR.glob("*.txt")}


@pytest.fixture
def minimal_profile_data() -> dict[str, Any]:
    """The smallest profile dict that validates. Tests mutate copies of this."""
    return {
        "wiki": {
            "id": "testwiki",
            "api": "https://example.test/api.php",
            "license": "CC BY-SA 4.0",
            "attribution": "Test Wiki",
            "language_filter": {"strategy": "none"},
        },
        "infobox": {
            "location": "inline",
            "template": "Infobox",
            "type_field": "kind",
            "type_map": {"widget": ["Item"]},
        },
    }
