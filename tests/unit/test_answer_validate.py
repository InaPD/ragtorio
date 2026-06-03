"""Citation validation: the check that makes "grounded" a claim rather than a hope.

The cases worth writing down are the ways a fabricated citation can look real - an id
one digit off, a citation of the graph when no graph facts were retrieved, a bracket
that is not a citation at all - and what the system does with the sentence that made
the claim.
"""

from __future__ import annotations

from ragtorio.answer.models import CitationCheck
from ragtorio.answer.validate import (
    citation_url,
    index_url_for,
    regeneration_note,
    unverified_claims,
    validate,
)
from ragtorio.index.models import Chunk, ChunkMatch
from ragtorio.retrieve.merge import merge
from ragtorio.retrieve.models import GraphResult, Route

INDEX_URL = "https://wiki.factorio.com/index.php"


def route() -> Route:
    return Route(question="what is it made of", intent="both")


def match(chunk_id: str = "factorio:1:0000", title: str = "Electronic circuit") -> ChunkMatch:
    return ChunkMatch(
        chunk=Chunk(
            chunk_id=chunk_id,
            wiki="factorio",
            page_id=1,
            title=title,
            revision_id=194123,
            section_path=("Crafting",),
            text="Circuits are assembled from iron plate and copper cable.",
        ),
        score=0.9,
    )


def context(*, passages: bool = True, graph: bool = False):
    return merge(
        route(),
        graph=GraphResult(template="recipe_tree", lines=("Iron plate: 1",)) if graph else None,
        passages=[match()] if passages else [],
    )


def test_a_citation_of_a_retrieved_chunk_resolves_to_its_revision():
    check = validate("Iron plate [factorio:1:0000].", context(), INDEX_URL)
    assert check.ok
    assert len(check.citations) == 1
    citation = check.citations[0]
    assert citation.title == "Electronic circuit"
    assert citation.section == "Crafting"
    assert citation.url.endswith("?title=Electronic_circuit&oldid=194123")


def test_an_id_that_was_never_retrieved_is_caught_however_plausible_it_looks():
    """One digit off is the realistic failure, not a wild invention."""
    check = validate("Iron plate [factorio:1:0001].", context(), INDEX_URL)
    assert not check.ok
    assert check.unknown_ids == ("factorio:1:0001",)


def test_the_offending_sentence_comes_back_so_a_retry_can_name_it():
    text = "Circuits need iron [factorio:1:0000]. They also need copper [factorio:9:0003]."
    check = validate(text, context(), INDEX_URL)
    assert check.offending_claims == ("They also need copper [factorio:9:0003].",)
    assert "[factorio:9:0003]" in regeneration_note(check)
    assert "They also need copper" in regeneration_note(check)


def test_citing_the_graph_when_no_graph_facts_were_retrieved_is_a_fabrication():
    check = validate("It costs 1 iron plate [graph].", context(graph=False), INDEX_URL)
    assert not check.ok
    assert check.unknown_ids == ("graph",)


def test_citing_the_graph_when_graph_facts_were_retrieved_is_recorded_as_such():
    check = validate("It costs 1 iron plate [graph].", context(graph=True), INDEX_URL)
    assert check.ok
    assert check.cited_graph


def test_bracketed_prose_is_not_mistaken_for_a_citation():
    check = validate("The wiki says [as of 1.1] that it is fine.", context(), INDEX_URL)
    assert check.ok
    assert check.citations == ()


def test_several_ids_in_one_bracket_are_all_checked():
    check = validate("Both agree [factorio:1:0000, factorio:1:0009].", context(), INDEX_URL)
    assert check.unknown_ids == ("factorio:1:0009",)
    assert [c.chunk_id for c in check.citations] == ["factorio:1:0000"]


def test_the_same_source_cited_twice_is_one_citation():
    text = "Iron [factorio:1:0000]. Copper [factorio:1:0000]."
    assert len(validate(text, context(), INDEX_URL).citations) == 1


def test_a_repeated_bad_id_is_reported_once():
    text = "Iron [factorio:2:0000]. Copper [factorio:2:0000]."
    assert validate(text, context(), INDEX_URL).unknown_ids == ("factorio:2:0000",)


def test_a_bulleted_list_of_facts_splits_into_one_claim_per_line():
    """Otherwise a whole list comes back as a single enormous "claim" and the
    regeneration prompt quotes the entire answer at the model."""
    text = "- Iron plate [factorio:1:0000]\n- Copper cable [factorio:7:0002]"
    check = validate(text, context(), INDEX_URL)
    assert check.offending_claims == ("- Copper cable [factorio:7:0002]",)


def test_unverified_falls_back_to_the_ids_when_no_sentence_can_be_quoted():
    check = CitationCheck(unknown_ids=("factorio:1:0009",))
    assert unverified_claims(check) == ("[factorio:1:0009]",)


def test_a_citation_url_pins_the_revision_and_escapes_the_title():
    url = citation_url(INDEX_URL, "Assembling machine 2", 12)
    assert url == "https://wiki.factorio.com/index.php?title=Assembling_machine_2&oldid=12"


def test_the_index_url_is_derived_from_the_profiles_api_url():
    assert index_url_for("https://wiki.factorio.com/api.php").endswith("/index.php")
    # A wiki serving the API from a subdirectory keeps it.
    assert index_url_for("https://x.test/w/api.php") == "https://x.test/w/index.php"
