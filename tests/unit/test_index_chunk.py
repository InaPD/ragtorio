"""Chunking, against the committed wikitext wherever a real fixture covers the case.

The synthetic cases here are the ones the six fixtures do not contain - a section over
the token budget, a section configured away - not restatements of what the real pages
already prove.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ragtorio.config import IndexConfig
from ragtorio.harvest.models import RawPage, RawRedirect
from ragtorio.index.chunk import Chunker, estimate_tokens
from ragtorio.ontology.canonical import TitleCanonicalizer

#: min_chars=0 so a one-line synthetic case still produces a chunk; the real profile's
#: floor gets its own test rather than silently swallowing every example here.
CONFIG = IndexConfig(
    mention_templates=["Icon", "icontech"],
    drop_sections=["See also", "History"],
    min_chars=0,
)


def page(wikitext: str, title: str = "Test page", page_id: int = 1) -> RawPage:
    return RawPage(
        wiki="factorio",
        page_id=page_id,
        ns=0,
        title=title,
        revision_id=7,
        revised_at=datetime(2026, 1, 1, tzinfo=UTC),
        wikitext=wikitext,
    )


@pytest.fixture
def chunker() -> Chunker:
    return Chunker("factorio", CONFIG)


@pytest.fixture
def oil_processing(wikitext: dict[str, str]) -> RawPage:
    return page(wikitext["Oil_processing"], "Oil processing")


def test_the_lead_section_keeps_an_empty_path(chunker: Chunker, oil_processing: RawPage):
    lead = chunker.chunk_page(oil_processing)[0]
    assert lead.section_path == ()
    assert lead.text.startswith("Oil processing is a large part of Factorio")


def test_subsections_carry_the_whole_heading_trail(chunker: Chunker, oil_processing: RawPage):
    paths = [chunk.section_path for chunk in chunker.chunk_page(oil_processing)]
    assert ("Setting up oil processing",) in paths
    assert ("Setting up oil processing", "Tips") in paths
    # A === subsection nests; the next == heading pops it rather than nesting deeper.
    assert ("Setting up oil processing", "Transporting fluids") in paths
    assert ("Optimal ratios",) in paths


def test_a_heading_reads_as_a_human_reads_it(chunker: Chunker, oil_processing: RawPage):
    """`=== Simple coal liquefaction {{SA}} ===` is a section about coal liquefaction."""
    paths = [chunk.section_path for chunk in chunker.chunk_page(oil_processing)]
    assert ("Setting up oil processing", "Simple coal liquefaction") in paths


def test_tables_are_dropped_from_the_text(chunker: Chunker, oil_processing: RawPage):
    overview = next(
        c for c in chunker.chunk_page(oil_processing) if c.section_path == ("Overview",)
    )
    assert "wikitable" not in overview.text
    assert "basic-oil-processing" not in overview.text


def test_but_the_tables_entities_still_count_as_mentioned(
    chunker: Chunker, oil_processing: RawPage
):
    """The Overview section is a recipe table: its prose never names petroleum gas."""
    overview = next(
        c for c in chunker.chunk_page(oil_processing) if c.section_path == ("Overview",)
    )
    assert "factorio:Petroleum gas" in overview.mentioned_entity_ids
    assert "factorio:Oil refinery" in overview.mentioned_entity_ids


def test_mentions_come_from_icon_templates_as_well_as_links(chunker: Chunker):
    chunk = chunker.chunk_page(page("A refinery needs {{Icon|Crude oil|100}} to run at all."))[0]
    assert chunk.mentioned_entity_ids == ("factorio:Crude oil",)


def test_mentions_are_deduplicated_and_keep_document_order(chunker: Chunker):
    text = "[[Crude oil]] becomes [[Heavy oil]], and more [[crude oil]] means more heavy oil."
    assert chunker.chunk_page(page(text))[0].mentioned_entity_ids == (
        "factorio:Crude oil",
        "factorio:Heavy oil",
    )


def test_files_categories_and_interwiki_links_are_not_entities(chunker: Chunker):
    text = (
        "[[File:Oil.png|thumb]] Crude oil is real, unlike [[:Wikipedia:Petroleum]] "
        "and [[Category:Fluids]]. See [[Heavy oil]] for the rest of it."
    )
    assert chunker.chunk_page(page(text))[0].mentioned_entity_ids == ("factorio:Heavy oil",)


def test_mentions_resolve_through_redirects_and_aliases():
    titles = TitleCanonicalizer(
        [RawRedirect(wiki="factorio", from_title="Green circuits", to_title="Electronic circuit")],
        {"green circuit": "Electronic circuit"},
    )
    chunker = Chunker("factorio", CONFIG, titles)
    text = "Both [[Green circuits]] and {{Icon|green circuit|10}} name the same thing here."
    assert chunker.chunk_page(page(text))[0].mentioned_entity_ids == (
        "factorio:Electronic circuit",
    )


def test_a_mention_the_graph_does_not_know_is_dropped():
    """`{{icon|time|5}}` names no entity, and neither does a red link."""
    chunker = Chunker("factorio", CONFIG, None, frozenset({"factorio:Crude oil"}))
    text = "It takes {{icon|time|5}} to turn [[Crude oil]] into [[Something invented]]."
    assert chunker.chunk_page(page(text))[0].mentioned_entity_ids == ("factorio:Crude oil",)


def test_configured_sections_are_skipped_with_everything_under_them(chunker: Chunker):
    text = (
        "The lead section says enough about this to be worth indexing on its own.\n\n"
        "== See also ==\n[[Oil processing]] and a list of other pages nobody asked for.\n\n"
        "=== Related ===\nMore links, still not prose, still not worth retrieving here.\n\n"
        "== Overview ==\nThis part is real prose and should survive the section filter.\n"
    )
    paths = [chunk.section_path for chunk in chunker.chunk_page(page(text))]
    assert paths == [(), ("Overview",)]


def test_a_section_over_the_budget_splits_on_paragraph_boundaries():
    paragraph = "Crude oil is refined in an oil refinery, and the output must be consumed. "
    text = "\n\n".join([paragraph * 6] * 4)
    chunks = Chunker("factorio", IndexConfig(max_tokens=100)).chunk_page(page(text))
    assert len(chunks) > 1
    assert all(estimate_tokens(chunk.text) <= 120 for chunk in chunks)
    # No text is lost and none is duplicated: the pieces reassemble the original.
    assert "".join("".join(c.text.split()) for c in chunks) == "".join(text.split())


def test_a_single_paragraph_over_the_budget_splits_on_sentences():
    text = " ".join(f"Sentence number {i} explains one more thing about oil." for i in range(40))
    chunks = Chunker("factorio", IndexConfig(max_tokens=100)).chunk_page(page(text))
    assert len(chunks) > 1
    assert all(chunk.text.endswith(".") for chunk in chunks)


def test_a_section_too_short_to_be_prose_is_not_indexed():
    """A one-line navigation stub is not a passage, whatever the chunker calls it."""
    text = (
        "Long enough to keep, this lead section describes what the page is about.\n\n"
        "== Nav ==\nSee below.\n"
    )
    chunks = Chunker("factorio", IndexConfig(min_chars=40)).chunk_page(page(text))
    assert [chunk.section_path for chunk in chunks] == [()]


def test_the_short_tail_of_a_long_section_is_kept():
    """The floor is a judgement about sections, not about the pieces of one: a split
    section's last paragraph is still prose somebody wrote."""
    text = "\n\n".join(["Crude oil is refined in an oil refinery. " * 8, "One short line."])
    chunks = Chunker("factorio", IndexConfig(max_tokens=100, min_chars=80)).chunk_page(page(text))
    assert chunks[-1].text.endswith("One short line.")


def test_chunk_ids_are_unique_within_a_page_and_carry_its_provenance(
    chunker: Chunker, oil_processing: RawPage
):
    chunks = chunker.chunk_page(oil_processing)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert all(chunk.chunk_id.startswith("factorio:1:") for chunk in chunks)
    assert all(chunk.revision_id == 7 and chunk.title == "Oil processing" for chunk in chunks)


def test_heading_renders_the_path_a_citation_shows(chunker: Chunker, oil_processing: RawPage):
    tips = next(c for c in chunker.chunk_page(oil_processing) if c.section_path[-1:] == ("Tips",))
    assert tips.heading == "Oil processing > Setting up oil processing > Tips"


def test_chunk_pages_walks_every_article(chunker: Chunker, factorio_pages: list[RawPage]):
    articles = [p for p in factorio_pages if p.ns == 0]
    chunks = chunker.chunk_pages(articles)
    assert len({chunk.page_id for chunk in chunks}) == len(articles)


def test_an_empty_page_yields_nothing(chunker: Chunker):
    assert chunker.chunk_page(page("")) == []
