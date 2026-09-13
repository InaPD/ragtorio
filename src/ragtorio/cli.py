"""Command-line entry point.

Phase 0 shipped ``probe`` and ``profile``; Phase 1 adds ``init-db`` and ``harvest``.
Later phases add extract, graph, index, ask and bench.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ragtorio.config import Settings, WikiProfile, load_profile
from ragtorio.db.connect import apply_schema, connect
from ragtorio.harvest.client import MediaWikiClient
from ragtorio.harvest.crawl import Crawler
from ragtorio.harvest.models import CrawlStats
from ragtorio.harvest.postgres import PostgresHarvestStore
from ragtorio.harvest.probe import ProbeResult
from ragtorio.harvest.probe import probe as run_probe
from ragtorio.harvest.store import HarvestStore, InMemoryHarvestStore

app = typer.Typer(
    help="Knowledge-graph and vector RAG over crafting-game wikis.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

VERDICT_STYLE = {
    "populated": "green",
    "installed_but_empty": "yellow",
    "template_only": "cyan",
}


@app.command()
def probe(
    api_url: str = typer.Argument(..., help="The wiki's api.php URL."),
    rps: float = typer.Option(1.0, help="Requests per second. Be polite."),
) -> None:
    """Inspect a wiki: extensions, namespaces, and whether structured data is real."""
    settings = Settings()
    _warn_on_default_contact(settings)
    with MediaWikiClient(api_url, settings.user_agent, rps=rps) as client:
        result = run_probe(client)
    _render_probe(result)


@app.command()
def profile(wiki_id: str = typer.Argument(..., help="Profile id, e.g. 'factorio'.")) -> None:
    """Validate a wiki profile and show what it will crawl."""
    loaded = _load_profile_or_exit(wiki_id)
    console.print(f"[green]OK[/] {wiki_id} profile is valid")
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("api", loaded.wiki.api)
    table.add_row("license", loaded.wiki.license)
    table.add_row("rate limit", f"{loaded.wiki.rate_limit_rps} req/s")
    table.add_row("crawl namespaces", str(loaded.crawl_namespaces))
    table.add_row("infobox", f"{loaded.infobox.location} (template {loaded.infobox.template})")
    table.add_row("mapped types", str(len(loaded.infobox.type_map)))
    table.add_row("mapped fields", str(len(loaded.infobox.fields)))
    ignored = [k for k, v in loaded.infobox.type_map.items() if not v]
    table.add_row("explicitly ignored", str(len(ignored)))
    console.print(table)


@app.command(name="init-db")
def init_db(
    dsn: str | None = typer.Option(None, help="Postgres DSN. Defaults to RAGTORIO_POSTGRES_DSN."),
) -> None:
    """Create the tables. Safe to run against an existing database."""
    target = dsn or Settings().postgres_dsn
    with connect(target) as conn:
        apply_schema(conn)
    console.print(f"[green]OK[/] schema applied to {_redact(target)}")


@app.command()
def harvest(
    wiki_id: str = typer.Argument(..., help="Profile id, e.g. 'factorio'."),
    limit: int | None = typer.Option(None, help="Stop after this many pages per namespace."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Crawl into memory, write nothing."),
    dsn: str | None = typer.Option(None, help="Postgres DSN. Defaults to RAGTORIO_POSTGRES_DSN."),
) -> None:
    """Fetch a wiki's pages, redirects and categories into Postgres."""
    settings = Settings()
    profile_data = _load_profile_or_exit(wiki_id)
    _warn_on_default_contact(settings)

    if dry_run:
        store: HarvestStore = InMemoryHarvestStore()
        stats = _crawl(profile_data, settings, store, limit)
        console.print("[yellow]dry run[/] nothing was written")
        _render_stats(stats, store.namespace_counts(wiki_id))
        return

    with connect(dsn or settings.postgres_dsn) as conn:
        apply_schema(conn)
        postgres_store = PostgresHarvestStore(conn)
        stats = _crawl(profile_data, settings, postgres_store, limit)
        _render_stats(stats, postgres_store.namespace_counts(wiki_id))


def _crawl(
    profile_data: WikiProfile, settings: Settings, store: HarvestStore, limit: int | None
) -> CrawlStats:
    """Run one crawl, reporting progress as each namespace completes."""
    with MediaWikiClient(
        profile_data.wiki.api,
        settings.user_agent,
        rps=profile_data.wiki.rate_limit_rps,
    ) as client:
        crawler = Crawler(client, profile_data, store, progress=lambda msg: console.print(msg))
        return crawler.run(limit=limit)


def _load_profile_or_exit(wiki_id: str) -> WikiProfile:
    """Load a profile, turning a missing file into a clean exit rather than a traceback."""
    try:
        return load_profile(wiki_id)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc


def _warn_on_default_contact(settings: Settings) -> None:
    """A crawl with no contact address is the thing wiki operators block."""
    if settings.contact_email == "you@example.com":
        console.print(
            "[yellow]warning:[/] RAGTORIO_CONTACT_EMAIL is unset, so the User-Agent has no real "
            "contact address. Set it in .env before any large crawl."
        )


def _redact(dsn: str) -> str:
    """Hide the password so a DSN can be printed or logged."""
    if "@" not in dsn or "//" not in dsn:
        return dsn
    scheme, _, rest = dsn.partition("//")
    credentials, _, host = rest.rpartition("@")
    user, sep, _ = credentials.partition(":")
    return f"{scheme}//{user}{sep}***@{host}" if sep else dsn


def _render_stats(stats: CrawlStats, stored: dict[int, int]) -> None:
    """Print the tally the phase exit criteria are checked against."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("listed", f"{stats.listed:,}")
    table.add_row("translations dropped", f"{stats.translations:,}")
    table.add_row("fetched", f"{stats.fetched:,}")
    table.add_row("unchanged", f"{stats.unchanged:,}")
    table.add_row("redirects", f"{stats.redirects:,}")
    console.print(table)

    if stats.by_namespace:
        console.print("\n[bold]pages kept per namespace[/]")
        for namespace, count in sorted(stats.by_namespace.items()):
            in_store = stored.get(namespace, 0)
            console.print(f"  {namespace:>5}  {count:>6,}  (in store: {in_store:,})")
    console.print()


def _render_probe(result: ProbeResult) -> None:
    """Print a probe result as a human would want to read it."""
    console.print(f"\n[bold]{result.api_url}[/]")

    overview = Table(show_header=False, box=None, pad_edge=False)
    overview.add_row("generator", result.generator)
    overview.add_row("articles", f"{result.articles:,}")
    overview.add_row("pages", f"{result.pages:,}")
    overview.add_row("license", result.license)
    overview.add_row(
        "page HTML",
        {True: "reachable", False: "blocked (use the API)", None: "unknown"}[
            result.html_accessible
        ],
    )
    console.print(overview)

    verdict = result.structured.verdict
    style = VERDICT_STYLE[verdict]
    extension = result.structured.extension or "none"
    console.print(f"\n[bold]structured data[/]  extension: {extension}")
    console.print(f"  [{style}]{verdict}[/]  {result.structured.detail}")

    if result.custom_namespaces:
        console.print("\n[bold]custom namespaces[/]")
        for ns_id, name in result.custom_namespaces:
            console.print(f"  {ns_id:>5}  {name}")

    console.print("\n[bold]language subpages[/]")
    if result.needs_language_filter:
        console.print(
            f"  [yellow]{result.language_subpage_ratio:.0%} of a 500-title sample[/] "
            f"e.g. {list(result.language_samples)}"
        )
        console.print("  profile needs language_filter.strategy: subpage_suffix")
    else:
        console.print("  none detected; language_filter.strategy: none")
    console.print()


if __name__ == "__main__":
    app()
