"""The extractor: infobox pages and their article's ``{{history}}`` templates to facts.

Walks the infobox namespace (not the mainspace) because that direction is the reliable
one: every ``Infobox:X`` page corresponds to exactly one article, ``X``, by construction
of the wiki's own namespace convention, whereas finding a page's infobox by parsing its
prose for a ``{{:Infobox:X}}`` transclusion would mean walking the much larger, far
more template-heavy mainspace for no extra information. ``{{history}}`` is the one
thing that lives only in the article, so that page is still fetched, but only by exact
title, once per infobox page.
"""

from __future__ import annotations

from collections import Counter

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
        if profile.infobox.location != "separate_namespace":
            raise ValueError(
                f"TemplateExtractor only supports infobox.location 'separate_namespace', "
                f"got {profile.infobox.location!r}"
            )
        assert profile.infobox.namespace_id is not None  # guaranteed by WikiProfile's own validator
        self._profile = profile
        self._pages = pages
        self._namespace_id = profile.infobox.namespace_id
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
        subject = _strip_namespace(infobox_page.title)
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

        article = self._pages.by_title(self.wiki, subject)
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
            return self._recipe_facts(subject, labels, raw, provenance)

        if mapping.parser == "factorio_recipe_expr":
            expr = _parse_recipe(raw)
            return [
                Fact(
                    subject=subject,
                    subject_labels=tuple(labels),
                    predicate=to,
                    object=ingredient.name,
                    props={"amount": ingredient.amount},
                    provenance=provenance,
                )
                for ingredient in expr.inputs
            ]

        if mapping.parser == "plus_list":
            names = _parse_list(raw)
            return [
                Fact(
                    subject=subject,
                    subject_labels=tuple(labels),
                    predicate=to,
                    object=name,
                    provenance=provenance,
                )
                for name in names
            ]

        return [
            Fact(
                subject=subject,
                subject_labels=tuple(labels),
                predicate=to,
                object=_coerce_scalar(raw, mapping.type),
                provenance=provenance,
            )
        ]

    def _recipe_facts(
        self, subject: str, labels: list[str], raw: str, provenance: Provenance
    ) -> list[Fact]:
        expr = _parse_recipe(raw)
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


def _parse_recipe(raw: str) -> RecipeExpr:
    result = PARSER_REGISTRY["factorio_recipe_expr"](raw)
    assert isinstance(result, RecipeExpr)  # the registry is keyed by name; this pins the type
    return result


def _parse_list(raw: str) -> list[str]:
    result = PARSER_REGISTRY["plus_list"](raw)
    assert isinstance(result, list)
    return result


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
