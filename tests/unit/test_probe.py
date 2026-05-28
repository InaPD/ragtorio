"""Probe logic, especially the installed-versus-populated distinction.

This is the test that encodes the project's founding discovery: the Factorio wiki
advertises Semantic MediaWiki and the Stardew Valley wiki advertises Cargo, and both
stores are empty. A probe that trusted the extension list would be worse than useless.
"""

from __future__ import annotations

from typing import Any

import httpx
import respx

from ragtorio.harvest.client import MediaWikiClient
from ragtorio.harvest.probe import probe

API = "https://example.test/api.php"
SITE_ROOT = "https://example.test/"

NAMESPACES = {
    "0": {"id": 0, "name": "", "canonical": ""},
    "102": {"id": 102, "name": "Property", "canonical": "Property"},
    "3002": {"id": 3002, "name": "Infobox"},
}


def siteinfo(extensions: list[str], namespaces: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "query": {
            "general": {"generator": "MediaWiki 1.43.9", "rights": "CC BY-NC-SA 3.0"},
            "statistics": {"articles": 5191, "pages": 16295},
            "namespaces": namespaces if namespaces is not None else NAMESPACES,
            "extensions": [{"name": name} for name in extensions],
            "rightsinfo": {"text": "CC BY-NC-SA 3.0"},
        }
    }


def allpages(titles: list[str]) -> dict[str, Any]:
    return {"query": {"allpages": [{"title": t} for t in titles]}}


def client() -> MediaWikiClient:
    return MediaWikiClient(API, "ua", rps=1000.0, sleep=lambda _: None)


@respx.mock
def test_smw_installed_but_only_builtin_properties() -> None:
    respx.get(SITE_ROOT).mock(return_value=httpx.Response(200))
    respx.get(API).mock(
        side_effect=[
            httpx.Response(200, json=siteinfo(["SemanticMediaWiki", "Scribunto"])),
            httpx.Response(
                200,
                json=allpages(
                    ["Property:Foaf:homepage", "Property:Owl:differentFrom", "Property:Dc:title"]
                ),
            ),
            httpx.Response(200, json=allpages(["Iron plate", "Iron gear wheel"])),
        ]
    )
    with client() as c:
        result = probe(c)
    assert result.structured.extension == "SemanticMediaWiki"
    assert result.structured.verdict == "installed_but_empty"
    assert "built-ins" in result.structured.detail


@respx.mock
def test_smw_with_real_properties_is_populated() -> None:
    respx.get(SITE_ROOT).mock(return_value=httpx.Response(200))
    respx.get(API).mock(
        side_effect=[
            httpx.Response(200, json=siteinfo(["SemanticMediaWiki"])),
            httpx.Response(200, json=allpages(["Property:Has recipe", "Property:Stack size"])),
            httpx.Response(200, json=allpages(["Iron plate"])),
        ]
    )
    with client() as c:
        result = probe(c)
    assert result.structured.verdict == "populated"


@respx.mock
def test_cargo_with_no_declared_tables_is_empty() -> None:
    respx.get(SITE_ROOT).mock(return_value=httpx.Response(200))
    respx.get(API).mock(
        side_effect=[
            httpx.Response(200, json=siteinfo(["Cargo", "Scribunto"])),
            httpx.Response(200, json={"cargotables": []}),
            httpx.Response(200, json=allpages(["Iron Bar"])),
        ]
    )
    with client() as c:
        result = probe(c)
    assert result.structured.extension == "Cargo"
    assert result.structured.verdict == "installed_but_empty"


@respx.mock
def test_cargo_with_tables_is_populated() -> None:
    respx.get(SITE_ROOT).mock(return_value=httpx.Response(200))
    respx.get(API).mock(
        side_effect=[
            httpx.Response(200, json=siteinfo(["Cargo"])),
            httpx.Response(200, json={"cargotables": ["Items", "Recipes"]}),
            httpx.Response(200, json=allpages(["Iron Bar"])),
        ]
    )
    with client() as c:
        result = probe(c)
    assert result.structured.verdict == "populated"


@respx.mock
def test_cargo_api_unavailable_is_handled() -> None:
    respx.get(SITE_ROOT).mock(return_value=httpx.Response(200))
    respx.get(API).mock(
        side_effect=[
            httpx.Response(200, json=siteinfo(["Cargo"])),
            httpx.Response(200, json={"error": {"code": "unknown_action", "info": "no"}}),
            httpx.Response(200, json=allpages(["Iron Bar"])),
        ]
    )
    with client() as c:
        result = probe(c)
    assert result.structured.verdict == "installed_but_empty"


@respx.mock
def test_no_structured_extension_means_template_only() -> None:
    respx.get(SITE_ROOT).mock(return_value=httpx.Response(200))
    respx.get(API).mock(
        side_effect=[
            httpx.Response(200, json=siteinfo(["Scribunto", "DynamicPageList3"])),
            httpx.Response(200, json=allpages(["Dirt"])),
        ]
    )
    with client() as c:
        result = probe(c)
    assert result.structured.extension is None
    assert result.structured.verdict == "template_only"


@respx.mock
def test_smw_without_property_namespace() -> None:
    respx.get(SITE_ROOT).mock(return_value=httpx.Response(200))
    respx.get(API).mock(
        side_effect=[
            httpx.Response(200, json=siteinfo(["SemanticMediaWiki"], namespaces={"0": {"id": 0}})),
            httpx.Response(200, json=allpages(["Thing"])),
        ]
    )
    with client() as c:
        result = probe(c)
    assert result.structured.verdict == "installed_but_empty"


@respx.mock
def test_language_subpages_are_detected() -> None:
    respx.get(SITE_ROOT).mock(return_value=httpx.Response(200))
    respx.get(API).mock(
        side_effect=[
            httpx.Response(200, json=siteinfo([])),
            httpx.Response(
                200,
                json=allpages(
                    ["Accumulator", "Accumulator/de", "Accumulator/ja", "Accumulator/pt-br"]
                ),
            ),
        ]
    )
    with client() as c:
        result = probe(c)
    assert result.language_subpage_ratio == 0.75
    assert result.needs_language_filter
    assert "Accumulator/de" in result.language_samples


@respx.mock
def test_no_language_subpages() -> None:
    respx.get(SITE_ROOT).mock(return_value=httpx.Response(200))
    respx.get(API).mock(
        side_effect=[
            httpx.Response(200, json=siteinfo([])),
            httpx.Response(200, json=allpages(["Iron Bar", "Keg", "Parsnip"])),
        ]
    )
    with client() as c:
        result = probe(c)
    assert result.language_subpage_ratio == 0.0
    assert not result.needs_language_filter


@respx.mock
def test_empty_mainspace_does_not_divide_by_zero() -> None:
    respx.get(SITE_ROOT).mock(return_value=httpx.Response(200))
    respx.get(API).mock(
        side_effect=[
            httpx.Response(200, json=siteinfo([])),
            httpx.Response(200, json={"query": {}}),
        ]
    )
    with client() as c:
        assert probe(c).language_subpage_ratio == 0.0


@respx.mock
def test_custom_namespaces_and_metadata_are_reported() -> None:
    respx.get(SITE_ROOT).mock(return_value=httpx.Response(200))
    respx.get(API).mock(
        side_effect=[
            httpx.Response(200, json=siteinfo([])),
            httpx.Response(200, json=allpages(["Thing"])),
        ]
    )
    with client() as c:
        result = probe(c)
    assert (3002, "Infobox") in result.custom_namespaces
    assert result.articles == 5191
    assert result.license == "CC BY-NC-SA 3.0"
    assert result.generator == "MediaWiki 1.43.9"
    assert result.html_accessible is True


@respx.mock
def test_blocked_html_is_reported() -> None:
    respx.get(SITE_ROOT).mock(return_value=httpx.Response(403))
    respx.get(API).mock(
        side_effect=[
            httpx.Response(200, json=siteinfo([])),
            httpx.Response(200, json=allpages(["Thing"])),
        ]
    )
    with client() as c:
        assert probe(c).html_accessible is False


@respx.mock
def test_unreachable_html_is_unknown_not_an_error() -> None:
    respx.get(SITE_ROOT).mock(side_effect=httpx.ConnectError("no route"))
    respx.get(API).mock(
        side_effect=[
            httpx.Response(200, json=siteinfo([])),
            httpx.Response(200, json=allpages(["Thing"])),
        ]
    )
    with client() as c:
        assert probe(c).html_accessible is None
