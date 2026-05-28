"""Download the wikitext fixtures the extractor tests run against.

Committed to the repo so the whole extraction layer is testable offline and in CI.
Re-run with ``make fixtures`` when the wiki changes shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from field_assistant.config import Settings, load_profile
from field_assistant.harvest.client import MediaWikiClient

OUT = Path("tests/fixtures/wikitext/factorio")

#: Chosen to cover every shape the parser must survive, not for tidiness.
PAGES = [
    "Iron gear wheel",  # simplest recipe
    "Infobox:Iron gear wheel",
    "Electronic circuit",  # multi-ingredient
    "Infobox:Electronic circuit",
    "Oil processing",  # fluid processing article (the real page;
    "Infobox:Uranium processing",  #   "Advanced oil processing" is a redirect)
    "Uranium processing",  # multi-output with fractional probabilities
    "Kovarex enrichment process",  # recipe whose output feeds its own input
    "Infobox:Kovarex enrichment process",
    "Automation (research)",  # technology: cost, allows, prerequisites
    "Infobox:Automation (research)",
    "Assembling machine 2",  # a crafting station
    "Infobox:Assembling machine 2",
    "Infobox:Iron axe",  # archived, no prototype-type, explicit "= output"
    "Infobox:Yumako tree",  # world entity, expected-resources, no recipe
]


def slugify(title: str) -> str:
    return title.replace(":", "__").replace(" ", "_").replace("(", "").replace(")", "")


def main() -> None:
    profile = load_profile("factorio")
    settings = Settings()
    OUT.mkdir(parents=True, exist_ok=True)

    with MediaWikiClient(
        profile.wiki.api, settings.user_agent, rps=profile.wiki.rate_limit_rps
    ) as client:
        data = client.query(
            prop="revisions", rvprop="content|ids", rvslots="main", titles="|".join(PAGES)
        )

    written = 0
    for page in data.get("pages", []):
        title = str(page.get("title", ""))
        revisions = page.get("revisions") or []
        if not revisions:
            print(f"  MISSING {title}")
            continue
        text = revisions[0]["slots"]["main"].get("content", "")
        path = OUT / f"{slugify(title)}.txt"
        path.write_text(text, encoding="utf-8")
        written += 1
        print(f"  {path}  ({len(text):,} chars, rev {revisions[0].get('revid')})")
    print(f"\nwrote {written}/{len(PAGES)} fixtures to {OUT}")


if __name__ == "__main__":
    main()
