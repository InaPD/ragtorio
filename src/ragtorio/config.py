"""Settings and wiki-profile loading.

A *wiki profile* is a YAML file under ``wikis/`` describing how to crawl one wiki and
how to turn its infobox templates into graph facts. Everything that differs between
wikis belongs there. Nothing in this module is specific to any one wiki.

Profiles are validated on load, so a typo in YAML fails immediately with a pointed
message rather than silently producing an empty graph three phases later.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Node labels the ontology allows. See IMPLEMENTATION_PLAN.md section 4.
KNOWN_LABELS: frozenset[str] = frozenset({"Item", "Fluid", "Recipe", "Station", "Unlock"})

#: Parser functions a profile may reference. Implemented in ``extract/parsers/`` (Phase 2);
#: listed here so that a misspelled parser name is caught when the profile loads.
KNOWN_PARSERS: frozenset[str] = frozenset(
    {
        "factorio_recipe_expr",
        "plus_list",
        "link_list",
        "name_template",
        "name_template_list",
        "duration",
    }
)

#: Grammar for a field mapping's ``to`` target.
_TO_PATTERN = re.compile(r"^(?:prop\.[a-z][a-z0-9_]*|rel\.[A-Z][A-Z0-9_]*|recipe)$")


class Settings(BaseSettings):
    """Runtime settings, read from the environment or a local ``.env``."""

    model_config = SettingsConfigDict(env_prefix="RAGTORIO_", env_file=".env", extra="ignore")

    contact_email: str = "you@example.com"
    postgres_dsn: str = "postgresql://ragtorio:ragtorio@localhost:5433/ragtorio"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "ragtorio"

    @property
    def user_agent(self) -> str:
        """Identify the crawler and give operators a way to reach us, as they expect."""
        return f"ragtorio/0.1 (wiki RAG research; contact: {self.contact_email})"


class LanguageFilter(BaseModel):
    """How to drop non-English pages.

    ``subpage_suffix`` drops titles whose final path segment is a language code
    (``Iron plate/de``). Wikis that put each language on its own domain use ``none``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Literal["subpage_suffix", "none"]
    keep: str | None = None

    @model_validator(mode="after")
    def _keep_required_for_suffix(self) -> Self:
        if self.strategy == "subpage_suffix" and not self.keep:
            raise ValueError("language_filter.keep is required when strategy is subpage_suffix")
        return self


class ArchivedConfig(BaseModel):
    """How the wiki marks content removed from the game."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    namespace_id: int | None = None
    category: str | None = None

    @property
    def is_configured(self) -> bool:
        return self.namespace_id is not None or self.category is not None


class WikiMeta(BaseModel):
    """Identity, access rules and licensing for one wiki."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    api: str
    license: str
    attribution: str
    rate_limit_rps: float = Field(default=1.0, gt=0, le=10)
    language_filter: LanguageFilter
    archived: ArchivedConfig = ArchivedConfig()
    extra_namespaces: list[int] = Field(default_factory=list)


class FieldMapping(BaseModel):
    """What one infobox parameter becomes.

    ``to`` is one of ``prop.<name>`` (a node property), ``rel.<NAME>`` (a relationship),
    or ``recipe`` (the inputs/outputs/time expression handled specially).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    to: str
    parser: str | None = None
    type: Literal["int", "float", "str"] | None = None

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not _TO_PATTERN.match(self.to):
            raise ValueError(
                f"field mapping target {self.to!r} must be 'prop.<snake_case>', "
                "'rel.<UPPER_CASE>' or 'recipe'"
            )
        if self.parser and self.parser not in KNOWN_PARSERS:
            raise ValueError(
                f"unknown parser {self.parser!r}; known parsers: {sorted(KNOWN_PARSERS)}"
            )
        if self.parser and self.type:
            raise ValueError("a field mapping takes either 'parser' or 'type', not both")
        return self


class InlineTemplate(BaseModel):
    """A template appearing in article body text rather than in the infobox."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    to: str
    args: list[str]


class TypeRule(BaseModel):
    """Fallback classification when ``type_field`` is absent from a page.

    Rules are tried in order and the first whose parameters are all present wins.
    Needed because 37 of Factorio's 590 infobox pages carry no ``prototype-type``:
    archived technologies and items, plus world entities.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    when_present: list[str] = Field(min_length=1)
    emit: list[str]

    @model_validator(mode="after")
    def _validate(self) -> Self:
        unknown = set(self.emit) - KNOWN_LABELS
        if unknown:
            raise ValueError(f"type rule emits unknown labels {sorted(unknown)}")
        return self

    def matches(self, present: set[str]) -> bool:
        return all(param in present for param in self.when_present)


class InfoboxConfig(BaseModel):
    """Where the infobox lives and how its parameters map onto the ontology."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    location: Literal["separate_namespace", "inline"]
    template: str
    type_field: str | None = None
    namespace_id: int | None = None
    type_map: dict[str, list[str]] = Field(default_factory=dict)
    type_rules: list[TypeRule] = Field(default_factory=list)
    fields: dict[str, FieldMapping] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.location == "separate_namespace" and self.namespace_id is None:
            raise ValueError("infobox.namespace_id is required when location is separate_namespace")
        if self.type_field and not self.type_map:
            raise ValueError("infobox.type_map must be non-empty when type_field is set")
        if not self.type_field and not self.type_rules:
            raise ValueError("infobox needs either type_field + type_map, or type_rules")
        for value, labels in self.type_map.items():
            unknown = set(labels) - KNOWN_LABELS
            if unknown:
                raise ValueError(
                    f"type_map[{value!r}] has unknown labels {sorted(unknown)}; "
                    f"known labels: {sorted(KNOWN_LABELS)}"
                )
        return self

    def labels_for(self, type_value: str) -> list[str]:
        """Labels for a ``type_field`` value. An empty list means deliberately ignored."""
        return self.type_map.get(type_value, [])

    def is_mapped(self, type_value: str) -> bool:
        """Whether the profile has an opinion about this type value at all."""
        return type_value in self.type_map

    def classify(self, type_value: str | None, present_params: set[str]) -> list[str] | None:
        """Labels for one infobox, or ``None`` if the profile cannot classify it.

        Prefers the explicit ``type_map``; falls back to ``type_rules`` when the page
        carries no type field. An empty list means deliberately ignored, which is a
        decision and therefore not ``None``.
        """
        if type_value is not None and type_value in self.type_map:
            return self.type_map[type_value]
        for rule in self.type_rules:
            if rule.matches(present_params):
                return rule.emit
        return None


class IndexConfig(BaseModel):
    """How article prose becomes retrievable chunks.

    Everything here is wiki grammar rather than policy: which namespace holds the
    articles, which templates name an entity inline (Factorio writes
    ``{{Icon|Crude oil|100}}`` far more often than it links crude oil), and which
    templates and sections are navigation furniture that would only dilute a chunk.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    namespace_id: int = 0
    max_tokens: int = Field(default=800, ge=100, le=4000)
    min_chars: int = Field(default=80, ge=0)
    mention_templates: list[str] = Field(default_factory=list)
    drop_sections: list[str] = Field(default_factory=list)
    exclude_title_prefixes: list[str] = Field(default_factory=list)

    @property
    def mention_template_names(self) -> frozenset[str]:
        """Mention templates, casefolded: the wiki writes both ``{{Icon}}`` and ``{{icon}}``."""
        return frozenset(name.casefold() for name in self.mention_templates)

    @property
    def dropped_section_names(self) -> frozenset[str]:
        """Section headings to skip, casefolded."""
        return frozenset(name.casefold() for name in self.drop_sections)

    def excludes(self, title: str) -> bool:
        """Whether a page is excluded from the index by its title.

        A last resort, for whole pages that are structurally not prose: a wiki's
        changelog archive is machine-generated, enormous, and answers no question a
        player asks. Prefixes rather than exact titles because these always come as a
        family (``Version history/1.1.0``, ``/1.2.0``, ...).
        """
        return any(title.startswith(prefix) for prefix in self.exclude_title_prefixes)


class GroundTruth(BaseModel):
    """Where the authoritative game data for benchmarking lives."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    path: str


class WikiProfile(BaseModel):
    """A complete description of one wiki. Loaded from ``wikis/<id>.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    wiki: WikiMeta
    infobox: InfoboxConfig
    inline_templates: dict[str, InlineTemplate] = Field(default_factory=dict)
    index: IndexConfig = IndexConfig()
    ground_truth: GroundTruth | None = None

    @property
    def crawl_namespaces(self) -> list[int]:
        """Mainspace plus every extra namespace the profile asks for, deduplicated."""
        namespaces = [0, *self.wiki.extra_namespaces]
        if self.wiki.archived.namespace_id is not None:
            namespaces.append(self.wiki.archived.namespace_id)
        return sorted(set(namespaces))


def profiles_dir() -> Path:
    """The repository's ``wikis/`` directory."""
    return Path(__file__).resolve().parents[2] / "wikis"


def load_profile(wiki_id: str, directory: Path | None = None) -> WikiProfile:
    """Load and validate ``wikis/<wiki_id>.yaml``.

    Raises:
        FileNotFoundError: if no profile exists for that id.
        pydantic.ValidationError: if the profile is malformed.
    """
    base = directory or profiles_dir()
    path = base / f"{wiki_id}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in base.glob("*.yaml") if not p.stem.endswith(".aliases"))
        raise FileNotFoundError(
            f"no profile at {path}. Available profiles: {available or '(none)'}"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return WikiProfile.model_validate(data)
