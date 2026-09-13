"""CLI wiring. The rendering logic is worth a smoke test; the logic beneath it is
covered elsewhere."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
import respx
from typer.testing import CliRunner

from ragtorio.cli import _redact, app
from ragtorio.harvest.client import MediaWikiClient

runner = CliRunner()
API = "https://example.test/api.php"
SITE_ROOT = "https://example.test/"


def _mock_wiki(*, extensions: list[str], titles: list[str]) -> None:
    respx.get(SITE_ROOT).mock(return_value=httpx.Response(200))
    respx.get(API).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "query": {
                        "general": {"generator": "MediaWiki 1.43.9"},
                        "statistics": {"articles": 100, "pages": 200},
                        "namespaces": {"3002": {"id": 3002, "name": "Infobox"}},
                        "extensions": [{"name": e} for e in extensions],
                        "rightsinfo": {"text": "CC BY-SA"},
                    }
                },
            ),
            httpx.Response(200, json={"query": {"allpages": [{"title": t} for t in titles]}}),
        ]
    )


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "probe" in result.stdout
    assert "profile" in result.stdout


@respx.mock
def test_probe_renders_template_only_verdict() -> None:
    _mock_wiki(extensions=["Scribunto"], titles=["Thing", "Thing/de"])
    result = runner.invoke(app, ["probe", API])
    assert result.exit_code == 0
    assert "template_only" in result.stdout
    assert "3002" in result.stdout
    assert "subpage_suffix" in result.stdout


@respx.mock
def test_probe_reports_no_language_filter_needed() -> None:
    _mock_wiki(extensions=[], titles=["Alpha", "Beta"])
    result = runner.invoke(app, ["probe", API])
    assert result.exit_code == 0
    assert "none detected" in result.stdout


def test_profile_shows_the_shipped_factorio_profile() -> None:
    result = runner.invoke(app, ["profile", "factorio"])
    assert result.exit_code == 0
    assert "valid" in result.stdout
    assert "wiki.factorio.com" in result.stdout


def test_profile_missing_exits_nonzero() -> None:
    result = runner.invoke(app, ["profile", "no-such-wiki"])
    assert result.exit_code == 1


FACTORIO_API = "https://wiki.factorio.com/api.php"


@respx.mock
def test_harvest_dry_run_crawls_without_a_database(
    monkeypatch: pytest.MonkeyPatch,
    fast_client: Callable[..., MediaWikiClient],
    factorio_responder: Callable[[httpx.Request], httpx.Response],
) -> None:
    monkeypatch.setattr("ragtorio.cli.MediaWikiClient", fast_client)
    respx.get(FACTORIO_API).mock(side_effect=factorio_responder)

    result = runner.invoke(app, ["harvest", "factorio", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "dry run" in result.output
    assert "fetched" in result.output


def test_harvest_with_an_unknown_profile_exits_cleanly() -> None:
    result = runner.invoke(app, ["harvest", "nosuchwiki", "--dry-run"])
    assert result.exit_code == 1
    assert "no profile at" in result.output


class TestRedact:
    def test_password_is_hidden(self) -> None:
        assert (
            _redact("postgresql://ragtorio:secret@localhost:5433/ragtorio")
            == "postgresql://ragtorio:***@localhost:5433/ragtorio"
        )

    def test_dsn_without_credentials_is_unchanged(self) -> None:
        assert (
            _redact("postgresql://localhost:5433/ragtorio")
            == "postgresql://localhost:5433/ragtorio"
        )

    def test_non_url_dsn_is_unchanged(self) -> None:
        assert _redact("dbname=ragtorio") == "dbname=ragtorio"
