"""Community shorthand a title never carries: "green circuit", "blue science".

A separate file from the profile itself (``wikis/<id>.aliases.yaml``) because this is
folklore, not wiki structure - it needs a human to notice a term is common and add it,
unlike the profile's field mappings which are derived mechanically from the infobox
grammar. Optional: a wiki with no such file simply gets no shorthand aliases.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ragtorio.config import profiles_dir


def load_aliases(wiki_id: str, directory: Path | None = None) -> dict[str, str]:
    """Shorthand to canonical title, e.g. ``{"green circuit": "Electronic circuit"}``.

    Returns an empty dict if the wiki has no aliases file.
    """
    base = directory or profiles_dir()
    path = base / f"{wiki_id}.aliases.yaml"
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    aliases = data.get("aliases", {})
    return {str(k): str(v) for k, v in aliases.items()}
