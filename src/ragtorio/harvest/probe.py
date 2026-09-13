"""Inspect a wiki before committing to it.

The question this answers is not "which extensions are installed" but "is there any
*populated* structured data here". Those differ: the Factorio wiki advertises Semantic
MediaWiki and the Stardew Valley wiki advertises Cargo, and both stores are empty. A
probe that reads the extension list alone would send you down a path that dead-ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from ragtorio.harvest.client import MediaWikiClient, MediaWikiError
from ragtorio.harvest.language import LANGUAGE_SUFFIX

#: Extensions that can hold typed data, as opposed to merely querying or templating it.
STRUCTURED_EXTENSIONS = ("SemanticMediaWiki", "Cargo")

#: Properties Semantic MediaWiki creates on install. Their presence proves nothing.
SMW_BUILTIN_PREFIXES = ("Foaf:", "Owl:", "Skos:", "Dc:", "Dct:", "Rdf:", "Rdfs:")

Verdict = Literal["template_only", "populated", "installed_but_empty"]


@dataclass(frozen=True)
class StructuredEvidence:
    """What the structured-data store actually contains."""

    extension: str | None
    verdict: Verdict
    detail: str


@dataclass(frozen=True)
class ProbeResult:
    """Everything worth knowing about a wiki before writing its profile."""

    api_url: str
    generator: str
    articles: int
    pages: int
    license: str
    extensions: tuple[str, ...]
    custom_namespaces: tuple[tuple[int, str], ...]
    structured: StructuredEvidence
    language_subpage_ratio: float
    language_samples: tuple[str, ...]
    html_accessible: bool | None

    @property
    def needs_language_filter(self) -> bool:
        """Translated subpages present in mainspace, so the profile must filter them."""
        return self.language_subpage_ratio > 0.05


def probe(client: MediaWikiClient) -> ProbeResult:
    """Run every check against one wiki and collect the findings."""
    info = client.query(
        meta="siteinfo",
        siprop="general|namespaces|statistics|extensions|rightsinfo",
    )
    general = info.get("general", {})
    statistics = info.get("statistics", {})
    namespaces = info.get("namespaces", {})
    extension_names = tuple(
        sorted({str(e.get("name") or e.get("namemsg") or "") for e in info.get("extensions", [])})
    )

    custom = tuple(
        (int(ns_id), str(ns.get("name", "")))
        for ns_id, ns in sorted(namespaces.items(), key=lambda kv: int(kv[0]))
        if int(ns_id) >= 100 and int(ns_id) % 2 == 0
    )

    # Order matters and is therefore explicit rather than left to argument evaluation
    # inside the constructor: structured evidence first, then the language sample.
    structured = _structured_evidence(client, extension_names, namespaces)
    ratio, samples = _language_evidence(client)

    return ProbeResult(
        api_url=client.api_url,
        generator=str(general.get("generator", "unknown")),
        articles=int(statistics.get("articles", 0)),
        pages=int(statistics.get("pages", 0)),
        license=str(info.get("rightsinfo", {}).get("text") or general.get("rights") or "unknown"),
        extensions=extension_names,
        custom_namespaces=custom,
        structured=structured,
        language_subpage_ratio=ratio,
        language_samples=samples,
        html_accessible=_html_accessible(client.api_url),
    )


def _structured_evidence(
    client: MediaWikiClient,
    extensions: tuple[str, ...],
    namespaces: dict[str, Any],
) -> StructuredEvidence:
    """Decide whether a structured store exists *and holds game data*."""
    if "SemanticMediaWiki" in extensions:
        return _smw_evidence(client, namespaces)
    if "Cargo" in extensions:
        return _cargo_evidence(client)
    return StructuredEvidence(
        extension=None,
        verdict="template_only",
        detail="no structured-data extension installed; parse infobox templates",
    )


def _smw_evidence(client: MediaWikiClient, namespaces: dict[str, Any]) -> StructuredEvidence:
    """Count user-defined SMW properties, ignoring the ones that ship with it."""
    property_ns = _namespace_id(namespaces, "Property")
    if property_ns is None:
        return StructuredEvidence(
            "SemanticMediaWiki", "installed_but_empty", "no Property namespace"
        )

    titles = [
        str(page.get("title", ""))
        for chunk in client.query_paged(list="allpages", apnamespace=property_ns, aplimit=500)
        for page in chunk.get("allpages", [])
    ]
    user_defined = [
        t for t in titles if not t.removeprefix("Property:").startswith(SMW_BUILTIN_PREFIXES)
    ]
    if user_defined:
        return StructuredEvidence(
            "SemanticMediaWiki",
            "populated",
            f"{len(user_defined)} user-defined properties, e.g. {user_defined[:5]}",
        )
    return StructuredEvidence(
        "SemanticMediaWiki",
        "installed_but_empty",
        f"{len(titles)} properties, all SMW built-ins; parse infobox templates instead",
    )


def _cargo_evidence(client: MediaWikiClient) -> StructuredEvidence:
    """Ask Cargo for its table list. No tables means nobody declared any."""
    try:
        body = client.request(action="cargotables")
    except (MediaWikiError, httpx.HTTPError):
        return StructuredEvidence("Cargo", "installed_but_empty", "cargotables API unavailable")

    tables = body.get("cargotables") or []
    if tables:
        return StructuredEvidence("Cargo", "populated", f"{len(tables)} tables, e.g. {tables[:5]}")
    return StructuredEvidence(
        "Cargo", "installed_but_empty", "0 declared tables; parse infobox templates instead"
    )


def _language_evidence(client: MediaWikiClient) -> tuple[float, tuple[str, ...]]:
    """Sample mainspace titles to see whether translations live as subpages."""
    body = client.query(list="allpages", apnamespace=0, aplimit=500, apfilterredir="nonredirects")
    titles = [str(p.get("title", "")) for p in body.get("allpages", [])]
    if not titles:
        return 0.0, ()
    translated = [t for t in titles if LANGUAGE_SUFFIX.search(t)]
    return len(translated) / len(titles), tuple(translated[:5])


def _html_accessible(api_url: str) -> bool | None:
    """Whether ordinary page HTML is reachable, or the API is the only way in.

    Best effort: a failure here is information, not an error.
    """
    parts = urlsplit(api_url)
    root = f"{parts.scheme}://{parts.netloc}/"
    try:
        response = httpx.get(
            root,
            headers={"User-Agent": "ragtorio/0.1 (access check)"},
            timeout=15.0,
            follow_redirects=True,
        )
    except httpx.HTTPError:
        return None
    return response.status_code < 400


def _namespace_id(namespaces: dict[str, Any], canonical: str) -> int | None:
    for ns_id, ns in namespaces.items():
        if ns.get("canonical") == canonical or ns.get("name") == canonical:
            return int(ns_id)
    return None
