"""Command-line entry point.

Phase 0 shipped ``probe`` and ``profile``; Phase 1 added ``init-db`` and ``harvest``;
Phase 2 added ``extract``; Phase 3 adds ``graph load``/``graph check`` and a
``--graph-only`` stopgap for ``ask``. Later phases add the vector index, real
routing and the benchmark.
"""

from __future__ import annotations

import re

import typer
from neo4j import Driver
from rich.console import Console
from rich.table import Table

from ragtorio.config import Settings, WikiProfile, load_profile
from ragtorio.db.connect import apply_schema, connect
from ragtorio.db.neo4j import apply_schema as apply_neo4j_schema
from ragtorio.db.neo4j import connect as neo4j_connect
from ragtorio.extract.base import ExtractionReport
from ragtorio.extract.postgres import PostgresFactStore
from ragtorio.extract.repository import PageRepository, PostgresPageRepository
from ragtorio.extract.store import FactStore
from ragtorio.extract.template import TemplateExtractor
from ragtorio.harvest.client import MediaWikiClient
from ragtorio.harvest.crawl import Crawler
from ragtorio.harvest.models import CrawlStats
from ragtorio.harvest.postgres import PostgresHarvestStore
from ragtorio.harvest.probe import ProbeResult
from ragtorio.harvest.probe import probe as run_probe
from ragtorio.harvest.store import HarvestStore, InMemoryHarvestStore
from ragtorio.ontology.aliases import load_aliases
from ragtorio.ontology.check import GraphCheckReport, check_graph
from ragtorio.ontology.load import GraphLoader
from ragtorio.ontology.models import ResolvedGraph
from ragtorio.ontology.recipe_tree import RecipeTreeNode, raw_totals, recipe_tree
from ragtorio.ontology.repository import PostgresGraphSourceRepository
from ragtorio.ontology.resolve import EntityResolver

app = typer.Typer(
    help="Knowledge-graph and vector RAG over crafting-game wikis.",
    no_args_is_help=True,
    add_completion=False,
)
graph_app = typer.Typer(help="Load facts into Neo4j and check the result.")
app.add_typer(graph_app, name="graph")
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


@app.command()
def extract(
    wiki_id: str = typer.Argument(..., help="Profile id, e.g. 'factorio'."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Read the crawl and report, write no facts."
    ),
    dsn: str | None = typer.Option(None, help="Postgres DSN. Defaults to RAGTORIO_POSTGRES_DSN."),
) -> None:
    """Turn a crawled wiki's infobox templates into fact rows.

    Reads ``raw_page`` (already there from ``ragtorio harvest``) and replaces every
    fact stored for this wiki, since a fact is always recomputable from the raw crawl.
    """
    profile_data = _load_profile_or_exit(wiki_id)

    with connect(dsn or Settings().postgres_dsn) as conn:
        apply_schema(conn)
        page_repository: PageRepository = PostgresPageRepository(conn)
        result = TemplateExtractor(profile_data, page_repository).extract()
        _render_report(result.report)

        if dry_run:
            console.print("[yellow]dry run[/] nothing was written")
            return

        fact_store: FactStore = PostgresFactStore(conn)
        fact_store.replace_all(wiki_id, result.facts)
        console.print(f"wrote {fact_store.count(wiki_id):,} facts")


@graph_app.command(name="load")
def graph_load(
    wiki_id: str = typer.Argument(..., help="Profile id, e.g. 'factorio'."),
    dsn: str | None = typer.Option(None, help="Postgres DSN. Defaults to RAGTORIO_POSTGRES_DSN."),
    neo4j_uri: str | None = typer.Option(None, help="Defaults to RAGTORIO_NEO4J_URI."),
) -> None:
    """Resolve a wiki's facts into a graph and load it into Neo4j.

    Idempotent: every node and edge is re-derived from ``fact`` and written with
    ``MERGE``, so running this again after a re-extraction is always safe.
    """
    settings = Settings()
    resolved = _resolve(wiki_id, dsn, settings)
    with _neo4j(settings, neo4j_uri) as driver:
        apply_neo4j_schema(driver)
        GraphLoader(driver).load(resolved)
    _render_resolution(resolved)


@graph_app.command(name="check")
def graph_check(
    wiki_id: str = typer.Argument(..., help="Profile id, e.g. 'factorio'."),
    dsn: str | None = typer.Option(None, help="Postgres DSN. Defaults to RAGTORIO_POSTGRES_DSN."),
    neo4j_uri: str | None = typer.Option(None, help="Defaults to RAGTORIO_NEO4J_URI."),
) -> None:
    """Check the loaded graph: orphans, incomplete recipes, self-cycles, unresolved
    references from the most recent resolution."""
    settings = Settings()
    resolved = _resolve(wiki_id, dsn, settings)
    with _neo4j(settings, neo4j_uri) as driver:
        report = check_graph(driver, resolved.unresolved)
    _render_check(report)


#: Crude, temporary question parsing. Real routing (entity extraction through an
#: LLM, resolved against aliases) is Phase 5; this exists only so the Phase 3 exit
#: criterion's exact command shape works today.
_GRAPH_ONLY_QUESTION = re.compile(
    r"raw (?:ore|materials?) (?:for|to (?:make|craft|produce)) "
    r"(?:one |a |an |\d+ )?(?P<entity>.+?)\.?$",
    re.IGNORECASE,
)


@app.command()
def ask(
    wiki_id: str = typer.Argument(..., help="Profile id, e.g. 'factorio'."),
    question: str = typer.Argument(..., help="A question, e.g. 'raw ore for one X'."),
    graph_only: bool = typer.Option(
        False, "--graph-only", help="Only implemented mode so far: recipe_tree."
    ),
    quantity: float = typer.Option(1.0, help="How many units of the entity."),
    neo4j_uri: str | None = typer.Option(None, help="Defaults to RAGTORIO_NEO4J_URI."),
) -> None:
    """Answer a question from the graph. Routing and vector retrieval are Phase 5/6;
    for now only ``--graph-only`` works, and only for a recipe_tree-shaped question."""
    if not graph_only:
        console.print(
            "[red]only --graph-only is implemented so far[/] "
            "(routing and vector retrieval are Phase 5/6)"
        )
        raise typer.Exit(code=1)

    match = _GRAPH_ONLY_QUESTION.search(question.strip())
    if match is None:
        console.print(f"[red]could not find an item name in:[/] {question!r}")
        raise typer.Exit(code=1)

    settings = Settings()
    with _neo4j(settings, neo4j_uri) as driver:
        title = _resolve_title(driver, wiki_id, match.group("entity").strip())
        if title is None:
            console.print(f"[red]no entity matching[/] {match.group('entity')!r}")
            raise typer.Exit(code=1)
        tree = recipe_tree(driver, wiki_id, title, quantity)
    _render_tree(tree)


def _resolve_title(driver: Driver, wiki: str, name: str) -> str | None:
    """A case-insensitive title lookup, standing in for Phase 5's real entity
    resolution through aliases."""
    with driver.session() as session:
        row = session.run(
            "MATCH (n) WHERE toLower(n.title) = toLower($name) AND n.id STARTS WITH $prefix "
            "RETURN n.title AS title LIMIT 1",
            name=name,
            prefix=f"{wiki}:",
        ).single()
    return str(row["title"]) if row else None


def _resolve(wiki_id: str, dsn: str | None, settings: Settings) -> ResolvedGraph:
    """Read facts and crawl metadata, and resolve them into a graph. Cheap and
    idempotent, so both ``graph load`` and ``graph check`` simply do it again."""
    profile_data = _load_profile_or_exit(wiki_id)
    with connect(dsn or settings.postgres_dsn) as conn:
        source = PostgresGraphSourceRepository(conn)
        facts = source.facts(wiki_id)
        redirects = source.redirects(wiki_id)
        archived = profile_data.wiki.archived
        archived_titles: frozenset[str] = frozenset()
        if archived.namespace_id is not None:
            archived_titles |= source.titles_in_namespace(wiki_id, archived.namespace_id)
        if archived.category is not None:
            archived_titles |= source.titles_in_category(wiki_id, archived.category)
    aliases = load_aliases(wiki_id)
    return EntityResolver(wiki_id, facts, redirects, aliases, archived_titles).resolve()


def _neo4j(settings: Settings, uri: str | None) -> Driver:
    """A driver using the given URI or the configured default, and the settings'
    Neo4j credentials either way."""
    return neo4j_connect(uri or settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)


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


def _render_report(report: ExtractionReport) -> None:
    """Print the coverage report the Phase 2 exit criteria are checked against."""
    coverage_style = "green" if report.coverage >= 0.95 else "yellow"
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("infobox pages seen", f"{report.pages_seen:,}")
    table.add_row("pages yielding facts", f"{report.pages_with_facts:,}")
    table.add_row("coverage", f"[{coverage_style}]{report.coverage:.1%}[/]")
    table.add_row("unclassified pages", f"{len(report.unclassified_pages):,}")
    table.add_row("parser failures", f"{len(report.parser_failures):,}")
    console.print(table)

    frequent = report.frequent_unknown_params()
    if frequent:
        console.print("\n[bold]unknown parameters seen more than 5 times[/]")
        for name, count in frequent:
            console.print(f"  {name:<30} {count:>4,}")

    if report.parser_failures:
        console.print("\n[bold]parser failures[/]")
        for failure in report.parser_failures:
            console.print(f"  {failure.page_title} / {failure.field}: {failure.error}")
            console.print(f"    value: {failure.value!r}")
    console.print()


def _render_resolution(resolved: ResolvedGraph) -> None:
    """Print what ``graph load`` wrote and what it could not resolve."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("nodes", f"{len(resolved.nodes):,}")
    table.add_row("edges", f"{len(resolved.edges):,}")
    table.add_row("unresolved references", f"{len(resolved.unresolved):,}")
    console.print(table)
    console.print()


def _render_check(report: GraphCheckReport) -> None:
    """Print the ``ragtorio graph check`` report."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("orphans", f"{len(report.orphans):,}")
    table.add_row("recipes missing inputs", f"{len(report.recipes_missing_inputs):,}")
    table.add_row("recipes missing outputs", f"{len(report.recipes_missing_outputs):,}")
    table.add_row("self-cycles (legitimate)", f"{len(report.self_cycles):,}")
    table.add_row("unresolved references", f"{len(report.unresolved):,}")
    console.print(table)

    if report.self_cycles:
        console.print("\n[bold]self-cycles[/] (a recipe consuming what it also produces)")
        for recipe_id, item_id in report.self_cycles:
            console.print(f"  {recipe_id}  <->  {item_id}")

    if report.orphans:
        console.print(f"\n[yellow]orphans[/] ({len(report.orphans)}, showing up to 10)")
        for node_id in report.orphans[:10]:
            console.print(f"  {node_id}")
    console.print()


def _render_tree(node: RecipeTreeNode) -> None:
    """Print a recipe tree as nested lines, then a raw-material total."""
    console.print(f"\n[bold]{node.title}[/]  x{node.amount:g}")
    for child in node.children:
        _render_node(child, indent="  ")

    console.print("\n[bold]raw materials[/]")
    for name, amount in sorted(raw_totals(node).items()):
        console.print(f"  {name}: {amount:g}")
    console.print()


def _render_node(node: RecipeTreeNode, indent: str) -> None:
    marker = " (cycle, truncated)" if node.truncated else " (raw)" if node.is_raw else ""
    console.print(f"{indent}{node.title}: {node.amount:g}{marker}")
    for child in node.children:
        _render_node(child, indent + "  ")


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
