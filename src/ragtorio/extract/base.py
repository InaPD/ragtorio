"""What any structured extractor must produce.

One method, deliberately: everything an extractor needs to decide internally (which
namespace to walk, how to parse one field) is its own business. A Lua-data or Cargo
extractor for a different wiki implements this same protocol without the caller
changing at all.
"""

from __future__ import annotations

from collections import Counter
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from ragtorio.extract.models import Fact


class ParserFailure(BaseModel):
    """One field that a named parser could not make sense of.

    Kept individually, with the offending value, because "recipe parsing failed" is
    useless without seeing what broke it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_title: str
    field: str
    value: str
    error: str


class ExtractionReport(BaseModel):
    """The coverage report ``ragtorio extract`` prints.

    ``pages_seen`` and ``pages_with_facts`` are what the Phase 2 exit criterion (95% of
    infobox pages yield at least one fact) is checked against. ``unknown_params`` is
    checked against the second criterion: every parameter seen more than 5 times
    should end up mapped or explicitly ignored in the profile.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pages_seen: int = 0
    pages_with_facts: int = 0
    unclassified_pages: tuple[str, ...] = ()
    unknown_params: Counter[str] = Field(default_factory=Counter)
    parser_failures: tuple[ParserFailure, ...] = ()

    @property
    def coverage(self) -> float:
        """Fraction of infobox pages that yielded at least one fact. 1.0 if none seen."""
        if self.pages_seen == 0:
            return 1.0
        return self.pages_with_facts / self.pages_seen

    def frequent_unknown_params(self, threshold: int = 5) -> list[tuple[str, int]]:
        """Unknown parameters seen more than ``threshold`` times, most frequent first.

        These are the ones the exit criterion says must be mapped or ignored; a
        parameter seen once or twice is very likely noise (a typo on one page) rather
        than something the profile needs an opinion about.
        """
        return sorted(
            ((name, count) for name, count in self.unknown_params.items() if count > threshold),
            key=lambda pair: pair[1],
            reverse=True,
        )


class ExtractionResult(BaseModel):
    """A complete extraction run: every fact produced, plus how it went."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    facts: tuple[Fact, ...]
    report: ExtractionReport


class StructuredExtractor(Protocol):
    """Produces facts from one wiki's structured data, however it stores it."""

    def extract(self) -> ExtractionResult: ...
