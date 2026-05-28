"""Command-line entry point.

Phase 0 ships ``ragtorio probe`` (inspect a wiki) and ``ragtorio profile`` (validate one we wrote).
Later phases add harvest, extract, graph, index, ask and bench.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ragtorio.config import Settings, load_profile
from ragtorio.harvest.client import MediaWikiClient
from ragtorio.harvest.probe import ProbeResult
from ragtorio.harvest.probe import probe as run_probe

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
    if settings.contact_email == "you@example.com":
        console.print(
            "[yellow]warning:[/] RAGTORIO_CONTACT_EMAIL is unset, so the User-Agent has no real "
            "contact address. Set it in .env before any large crawl."
        )
    with MediaWikiClient(api_url, settings.user_agent, rps=rps) as client:
        result = run_probe(client)
    _render_probe(result)


@app.command()
def profile(wiki_id: str = typer.Argument(..., help="Profile id, e.g. 'factorio'.")) -> None:
    """Validate a wiki profile and show what it will crawl."""
    try:
        loaded = load_profile(wiki_id)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

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
