"""The extractor: infobox templates and inline ``{{history}}`` templates to facts.

Two layouts, because wikis disagree about where an infobox lives.

**``separate_namespace``** is Factorio's: every ``Infobox:X`` page corresponds to
exactly one article, ``X``, by construction of the wiki's own namespace convention.
Walking that namespace is the reliable direction - finding a page's infobox by parsing
its prose for a ``{{:Infobox:X}}`` transclusion would mean walking the much larger, far
more template-heavy mainspace for no extra information. ``{{history}}`` is the one
thing that lives only in the article, so that page is fetched separately, by exact
title, once per infobox page.

**``inline``** is what most wikis do: the infobox is a template call in the article
itself. The walk is then over the article namespace, the subject is the page's own
title, and there is no second page to fetch - the article being walked is already the
one whose ``{{history}}`` templates matter.

Everything between those two points is shared, which is the reason the difference is
two branches rather than two extractors.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import mwparserfromhell
from mwparserfromhell.nodes import Template

from ragtorio.config import FieldMapping, WikiProfile
from ragtorio.extract.base import ExtractionReport, ExtractionResult, ParserFailure
from ragtorio.extract.models import Fact, Provenance
from ragtorio.extract.parsers import PARSER_REGISTRY
from ragtorio.extract.parsers.recipe import Ingredient, RecipeExpr
from ragtorio.extract.repository import PageRepository
from ragtorio.harvest.models import RawPage

#: The field name is a sentinel in ``FieldMapping.to``, not a real parser target: it
#: means "parse the recipe grammar and derive a Recipe's time, inputs and outputs",
#: rather than "write one property or one relationship".
_RECIPE_SENTINEL = "recipe"


class TemplateExtractor:
    """Extracts one wiki's facts according to its profile.

    Constructing this validates every parser name the profile references against
    :data:`~ragtorio.extract.parsers.PARSER_REGISTRY`, so a typo or a parser reserved
    for a wiki not yet implemented fails immediately rather than mid-crawl.
    """

    def __init__(self, profile: WikiProfile, pages: PageRepository) -> None:
        self._profile = profile
        self._pages = pages
        self._separate = profile.infobox.location == "separate_namespace"
        self._namespace_id = profile.infobox.source_namespace
        self._known_params = _known_params(profile)
        _validate_parsers(profile)

    @property
    def wiki(self) -> str:
        return self._profile.wiki.id

    def extract(self) -> ExtractionResult:
        facts: list[Fact] = []
        pages_seen = 0
        pages_with_facts = 0
        unclassified: list[str] = []
        unknown_params: Counter[str] = Counter()
        failures: list[ParserFailure] = []

        for infobox_page in self._pages.by_namespace(self.wiki, self._namespace_id):
            pages_seen += 1
            outcome = self._extract_page(infobox_page, unknown_params, failures)
            if outcome is None:
                unclassified.append(infobox_page.title)
                continue
            if outcome:
                pages_with_facts += 1
            facts.extend(outcome)

        report = ExtractionReport(
            pages_seen=pages_seen,
            pages_with_facts=pages_with_facts,
            unclassified_pages=tuple(unclassified),
            unknown_params=unknown_params,
            parser_failures=tuple(failures),
        )
        return ExtractionResult(facts=tuple(facts), report=report)

    def _extract_page(
        self,
        infobox_page: RawPage,
        unknown_params: Counter[str],
        failures: list[ParserFailure],
    ) -> list[Fact] | None:
        """Facts for one infobox page, or ``None`` if the profile cannot classify it."""
        template = _find_template(infobox_page.wikitext, self._profile.infobox.template)
        if template is None:
            return []

        params = {str(p.name).strip(): str(p.value).strip() for p in template.params}
        # Only strip a prefix when there is one to strip: an inline wiki's titles can
        # legitimately contain a colon ("Tutorial:Circuit network cookbook").
        subject = _strip_namespace(infobox_page.title) if self._separate else infobox_page.title
        labels = self._classify(params)
        if labels is None:
            return None
        if not labels:
            return []  # deliberately ignored: seen, but not a fact-yielding page

        provenance = Provenance(
            wiki=self.wiki,
            page_id=infobox_page.page_id,
            revision_id=infobox_page.revision_id,
            field="",
        )
        facts: list[Fact] = []
        for field_name, mapping in self._profile.infobox.fields.items():
            raw = params.get(field_name)
            if not raw:
                continue
            field_provenance = provenance.model_copy(update={"field": field_name})
            try:
                facts.extend(self._facts_for_field(subject, labels, mapping, raw, field_provenance))
            except ValueError as exc:
                failures.append(
                    ParserFailure(page_title=subject, field=field_name, value=raw, error=str(exc))
                )

        for name in params:
            if name not in self._known_params:
                unknown_params[name] += 1

        # Inline: the page being walked is the article, so there is nothing to fetch.
        article = infobox_page if not self._separate else self._pages.by_title(self.wiki, subject)
        if article is not None:
            facts.extend(self._history_facts(subject, labels, article))

        return facts

    def _classify(self, params: dict[str, str]) -> list[str] | None:
        infobox = self._profile.infobox
        type_value = params.get(infobox.type_field) if infobox.type_field else None
        return infobox.classify(type_value or None, set(params))

    def _facts_for_field(
        self,
        subject: str,
        labels: list[str],
        mapping: FieldMapping,
        raw: str,
        provenance: Provenance,
    ) -> list[Fact]:
        to = mapping.to
        if to == _RECIPE_SENTINEL:
            # mapping.parser, not a hardcoded name: WikiProfile guarantees the recipe
            # target has one, and which grammar it speaks is the whole point.
            assert mapping.parser is not None
            return self._recipe_facts(subject, labels, mapping.parser, raw, provenance)

        def fact(value: str | float, **props: Any) -> Fact:
            return Fact(
                subject=subject,
                subject_labels=tuple(labels),
                predicate=to,
                object=value,
                props=props,
                provenance=provenance,
            )

        if mapping.parser is None:
            return [fact(_coerce_scalar(raw, mapping.type))]

        # Dispatch on what the parser returns, not on which parser it is. Keying this
        # on the parser's name meant a new one silently fell through to the scalar
        # branch and its whole list arrived as a single unsplit string - which is how
        # `allows` produced one fact reading "Flammables + Plastics + Sulfur
        # processing" instead of three edges.
        parsed = _run_parser(mapping.parser, raw)
        if isinstance(parsed, RecipeExpr):
            return [fact(i.name, amount=i.amount) for i in parsed.inputs]
        if isinstance(parsed, list):
            return [fact(str(name)) for name in parsed]
        return [fact(_coerce_scalar(str(parsed), mapping.type))]

    def _recipe_facts(
        self,
        subject: str,
        labels: list[str],
        parser: str,
        raw: str,
        provenance: Provenance,
    ) -> list[Fact]:
        expr = _parse_recipe(parser, raw)
        facts = [
            Fact(
                subject=subject,
                subject_labels=tuple(labels),
                predicate="prop.crafting_time",
                object=expr.time,
                provenance=provenance,
            )
        ]
        facts.extend(
            Fact(
                subject=subject,
                subject_labels=tuple(labels),
                predicate="rel.CONSUMES",
                object=ingredient.name,
                props={"amount": ingredient.amount},
                provenance=provenance,
            )
            for ingredient in expr.inputs
        )
        outputs = expr.outputs or (Ingredient(name=subject, amount=1.0),)
        facts.extend(
            Fact(
                subject=subject,
                subject_labels=tuple(labels),
                predicate="rel.PRODUCES",
                object=ingredient.name,
                props={"amount": ingredient.amount, "probability": ingredient.probability},
                provenance=provenance,
            )
            for ingredient in outputs
        )
        return facts

    def _history_facts(self, subject: str, labels: list[str], article: RawPage) -> list[Fact]:
        facts = []
        for mapping_name, mapping in self._profile.inline_templates.items():
            for template in _find_templates(article.wikitext, mapping_name):
                values = [str(p.value).strip() for p in template.params]
                if len(values) < len(mapping.args):
                    continue
                event = dict(zip(mapping.args, values, strict=False))
                facts.append(
                    Fact(
                        subject=subject,
                        subject_labels=tuple(labels),
                        predicate=f"prop.{mapping.to}",
                        object=None,
                        props=event,
                        provenance=Provenance(
                            wiki=self.wiki,
                            page_id=article.page_id,
                            revision_id=article.revision_id,
                            field=mapping_name,
                        ),
                    )
                )
        return facts


def _known_params(profile: WikiProfile) -> frozenset[str]:
    """Every infobox parameter name the profile has an opinion about.

    That is the type field itself, every mapped field, and every parameter a
    ``type_rule`` checks for -- ``allows`` is deliberately never mapped to an edge (it
    is the exact inverse of ``required-technologies``) but it is still what tells the
    profile a page is a technology, so it must not show up as "unknown".
    """
    known = set(profile.infobox.fields)
    if profile.infobox.type_field:
        known.add(profile.infobox.type_field)
    for rule in profile.infobox.type_rules:
        known.update(rule.when_present)
    return frozenset(known)


def _validate_parsers(profile: WikiProfile) -> None:
    """Fail at construction time, not mid-crawl, on a parser name with no implementation."""
    names = {m.parser for m in profile.infobox.fields.values() if m.parser}
    unimplemented = names - PARSER_REGISTRY.keys()
    if unimplemented:
        raise ValueError(
            f"profile references parsers with no implementation yet: {sorted(unimplemented)}"
        )


def _parse_recipe(parser: str, raw: str) -> RecipeExpr:
    """Run the profile's recipe parser and insist it produced a recipe.

    The ``recipe`` target is the one field mapping whose *shape* the ontology depends
    on - a time, inputs, and optional outputs - so a parser wired to it that returns
    something else is a profile error worth naming rather than a crash three frames
    later.
    """
    result = PARSER_REGISTRY[parser](raw)
    if not isinstance(result, RecipeExpr):
        raise ValueError(
            f"the 'recipe' target needs a parser returning a RecipeExpr; "
            f"{parser!r} returned {type(result).__name__}"
        )
    return result


def _run_parser(name: str, raw: str) -> Any:
    """Run a profile-named parser. Construction already proved it is implemented."""
    return PARSER_REGISTRY[name](raw)


def _strip_namespace(title: str) -> str:
    """``Infobox:Iron gear wheel`` to ``Iron gear wheel``."""
    _, _, rest = title.partition(":")
    return rest if rest else title


def _find_template(wikitext: str, name: str) -> Template | None:
    """The first template call matching ``name``, case-insensitively.

    MediaWiki capitalises a page's own first letter but not a template's, and this
    wiki's infobox pages are inconsistent about it (``{{Infobox`` and ``{{infobox``
    both occur), so an exact match would silently drop real pages.
    """
    for template in _find_templates(wikitext, name):
        return template
    return None


def _find_templates(wikitext: str, name: str) -> list[Template]:
    code = mwparserfromhell.parse(wikitext)
    target = name.strip().lower()
    return [
        t for t in code.filter_templates(recursive=True) if str(t.name).strip().lower() == target
    ]


def _coerce_scalar(raw: str, scalar_type: str | None) -> str | float:
    if scalar_type == "int":
        try:
            return float(int(raw))
        except ValueError as exc:
            raise ValueError(f"not an integer: {raw!r}") from exc
    if scalar_type == "float":
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"not a number: {raw!r}") from exc
    return raw
