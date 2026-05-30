"""The coverage report's two exit-criteria calculations."""

from __future__ import annotations

from ragtorio.extract.base import ExtractionReport


def test_coverage_is_the_fraction_of_pages_yielding_facts() -> None:
    assert ExtractionReport(pages_seen=10, pages_with_facts=9).coverage == 0.9
    assert ExtractionReport().coverage == 1.0  # nothing seen: nothing to fail on


def test_frequent_unknown_params_excludes_the_exactly_at_threshold_case() -> None:
    """'more than 5', not 'at least 5': the exit criterion's own wording."""
    report = ExtractionReport(unknown_params={"category": 7, "borderline": 5, "rare": 1})
    assert report.frequent_unknown_params() == [("category", 7)]
