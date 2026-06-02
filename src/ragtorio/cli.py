"""Command-line entry point.

Phase 0 shipped ``probe`` and ``profile``; Phase 1 added ``init-db`` and ``harvest``;
Phase 2 added ``extract``; Phase 3 added ``graph load``/``graph check`` and a
``--graph-only`` stopgap for ``ask``; Phase 4 added ``index build``/``index recall``;
Phase 5 replaces that stopgap with real routing and adds ``route eval``. Phase 6 adds
grounded answering and the API.
"""

from __future__ import annotations

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
from ragtorio.index.build import IndexReport, build_index
from ragtorio.index.embed import EmbeddingProvider, build_provider
from ragtorio.index.postgres import PostgresChunkStore, ensure_embedding_dimension
from ragtorio.index.recall import RecallReport, evaluate, load_queries, sweep
from ragtorio.index.repository import PostgresChunkSourceRepository
from ragtorio.index.store import ChunkStore
from ragtorio.ontology.aliases import load_aliases
from ragtorio.ontology.check import GraphCheckReport, check_graph
from ragtorio.ontology.load import GraphLoader
from ragtorio.ontology.models import ResolvedGraph
from ragtorio.ontology.repository import PostgresGraphSourceRepository
from ragtorio.ontology.resolve import EntityResolver
from ragtorio.retrieve.entities import GraphEntityResolver
from ragtorio.retrieve.evaluate import RouterReport, evaluate_router, load_questions
from ragtorio.retrieve.graph import GraphRetriever
from ragtorio.retrieve.log import PostgresRoutingLog, RoutingLog
from ragtorio.retrieve.models import Intent, RetrievedContext
from ragtorio.retrieve.pipeline import RetrievalPipeline
from ragtorio.retrieve.router import AnthropicRouter, RouterUnavailableError
from ragtorio.retrieve.vector import VectorRetriever

app = typer.Typer(
    help="Knowledge-graph and vector RAG over crafting-game wikis.",
    no_args_is_help=True,
    add_completion=False,
)
graph_app = typer.Typer(help="Load facts into Neo4j and check the result.")
app.add_typer(graph_app, name="graph")
index_app = typer.Typer(help="Chunk articles into a vector index and measure its recall.")
app.add_typer(index_app, name="index")
route_app = typer.Typer(help="Measure the question router against its labeled set.")
app.add_typer(route_app, name="route")
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


@index_app.command(name="build")
def index_build(
    wiki_id: str = typer.Argument(..., help="Profile id, e.g. 'factorio'."),
    limit: int | None = typer.Option(None, help="Stop after this many articles."),
    provider: str = typer.Option(
        "sentence-transformers", help="sentence-transformers | voyage | hashing."
    ),
    model: str | None = typer.Option(None, help="Model name. Defaults to the provider's."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Chunk and embed, write nothing."),
    dsn: str | None = typer.Option(None, help="Postgres DSN. Defaults to RAGTORIO_POSTGRES_DSN."),
) -> None:
    """Split a wiki's articles into chunks, embed them, and store them in pgvector.

    Reads ``raw_page`` (already there from ``ragtorio harvest``) and replaces every
    chunk stored for this wiki, since a chunk is always recomputable from the raw
    crawl. Entity mentions are filtered against the titles that produced facts, so
    running ``ragtorio extract`` first gives a more useful index.
    """
    profile_data = _load_profile_or_exit(wiki_id)
    embedder = _build_provider_or_exit(provider, model)

    with connect(dsn or Settings().postgres_dsn) as conn:
        apply_schema(conn)
        source = PostgresChunkSourceRepository(conn)
        result = build_index(
            profile_data, source, embedder, limit=limit, progress=lambda msg: console.print(msg)
        )
        _render_index_report(result.report)

        if dry_run:
            console.print("[yellow]dry run[/] nothing was written")
            return

        if ensure_embedding_dimension(conn, embedder.dimensions):
            console.print(f"chunk.embedding resized to vector({embedder.dimensions})")
        store: ChunkStore = PostgresChunkStore(conn)
        store.replace_all(wiki_id, result.chunks)
        console.print(f"wrote {store.count(wiki_id):,} chunks")


@index_app.command(name="recall")
def index_recall(
    wiki_id: str = typer.Argument(..., help="Profile id, e.g. 'factorio'."),
    provider: str = typer.Option(
        "sentence-transformers", help="Must match the provider the index was built with."
    ),
    model: str | None = typer.Option(None, help="Model name. Defaults to the provider's."),
    ef_search: str | None = typer.Option(
        None, "--ef-search", help="Comma-separated hnsw.ef_search values to compare, e.g. 40,100."
    ),
    show_misses: bool = typer.Option(False, "--show-misses", help="List every query that missed."),
    dsn: str | None = typer.Option(None, help="Postgres DSN. Defaults to RAGTORIO_POSTGRES_DSN."),
) -> None:
    """Measure recall@k against the wiki's hand-labeled query set.

    With ``--ef-search`` this sweeps the HNSW accuracy setting, which is how the value
    to use in production gets chosen: recall rises and latency with it, and the right
    trade-off is a property of this index rather than a number to copy.
    """
    embedder = _build_provider_or_exit(provider, model)
    try:
        queries = load_queries(wiki_id)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    with connect(dsn or Settings().postgres_dsn) as conn:
        apply_schema(conn)
        store: ChunkStore = PostgresChunkStore(conn)
        if store.count(wiki_id) == 0:
            console.print(f"[red]no chunks stored for {wiki_id}[/] - run `ragtorio index build`")
            raise typer.Exit(code=1)
        indexed = store.titles(wiki_id)
        if ef_search:
            reports = sweep(
                store, embedder, wiki_id, queries, _int_list(ef_search), indexed_titles=indexed
            )
            _render_sweep(reports)
            return
        report = evaluate(store, embedder, wiki_id, queries, indexed_titles=indexed)
    _render_recall(report, show_misses)


@app.command()
def ask(
    wiki_id: str = typer.Argument(..., help="Profile id, e.g. 'factorio'."),
    question: str = typer.Argument(..., help="A question about the wiki."),
    intent: str | None = typer.Option(
        None, help="Force graph | vector | both, bypassing the router's own choice."
    ),
    provider: str = typer.Option(
        "sentence-transformers", help="Must match the provider the index was built with."
    ),
    model: str | None = typer.Option(None, help="Embedding model. Defaults to the provider's."),
    show_context: bool = typer.Option(
        False, "--show-context", help="Print the retrieved passages, not just the route."
    ),
    dsn: str | None = typer.Option(None, help="Postgres DSN. Defaults to RAGTORIO_POSTGRES_DSN."),
    neo4j_uri: str | None = typer.Option(None, help="Defaults to RAGTORIO_NEO4J_URI."),
) -> None:
    """Route a question, retrieve from the graph and the index, and show the context.

    This is retrieval only. Generating a grounded answer from the context, with
    citations validated against the chunk ids below, is Phase 6.
    """
    settings = Settings()
    profile_data = _load_profile_or_exit(wiki_id)
    forced = _intent_or_exit(intent)
    embedder = _build_provider_or_exit(provider, model)

    with (
        connect(dsn or settings.postgres_dsn) as conn,
        _neo4j(settings, neo4j_uri) as driver,
    ):
        apply_schema(conn)
        chunk_store: ChunkStore = PostgresChunkStore(conn)
        log: RoutingLog = PostgresRoutingLog(conn)
        pipeline = RetrievalPipeline(
            wiki_id,
            router=_router_or_exit(driver, wiki_id),
            graph=GraphRetriever(driver, wiki_id, raw_items=profile_data.resolution.raw_item_set),
            vector=VectorRetriever(chunk_store, embedder, wiki_id),
            log=log,
        )
        try:
            context = pipeline.retrieve(question, force_intent=forced)
        except RouterUnavailableError as exc:
            raise _exit_on_unavailable_router(exc) from exc
    _render_context(context, show_context)


@route_app.command(name="eval")
def route_eval(
    wiki_id: str = typer.Argument(..., help="Profile id, e.g. 'factorio'."),
    show_failures: bool = typer.Option(
        False, "--show-failures", help="List every question the router got wrong."
    ),
    neo4j_uri: str | None = typer.Option(None, help="Defaults to RAGTORIO_NEO4J_URI."),
) -> None:
    """Measure the router against the wiki's hand-labeled question set.

    One model call per question. Entities are resolved against the loaded graph, so
    this also reports the mentions the router invented or spelled in a way nothing
    matches - which an accuracy number on its own would hide.
    """
    settings = Settings()
    try:
        questions = load_questions(wiki_id)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    with _neo4j(settings, neo4j_uri) as driver:
        try:
            report = evaluate_router(_router_or_exit(driver, wiki_id), questions)
        except RouterUnavailableError as exc:
            raise _exit_on_unavailable_router(exc) from exc
    _render_router_report(report, show_failures)


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
        categories = source.categories_by_title(wiki_id)
    aliases = load_aliases(wiki_id)
    return EntityResolver(
        wiki_id,
        facts,
        redirects,
        aliases,
        archived_titles,
        categories,
        profile_data.resolution.reference_suffixes,
        profile_data.resolution.recipe_suffix,
    ).resolve()


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


def _build_provider_or_exit(provider: str, model: str | None) -> EmbeddingProvider:
    """Construct an embedding provider, turning a bad name or a missing dependency
    into a clean exit rather than a traceback."""
    try:
        return build_provider(provider, model)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc


def _int_list(raw: str) -> list[int]:
    """Parse a comma-separated option like ``40,100,200``."""
    try:
        return [int(part) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        console.print(f"[red]expected comma-separated integers, got[/] {raw!r}")
        raise typer.Exit(code=1) from exc


def _intent_or_exit(intent: str | None) -> Intent | None:
    """Validate a forced intent, which is how the benchmark's two baselines are run."""
    if intent is None:
        return None
    allowed: dict[str, Intent] = {"graph": "graph", "vector": "vector", "both": "both"}
    if intent in allowed:
        return allowed[intent]
    console.print(f"[red]intent must be graph, vector or both, got[/] {intent!r}")
    raise typer.Exit(code=1)


def _router_or_exit(driver: Driver, wiki_id: str) -> AnthropicRouter:
    """Build the router. Credentials are not checked here - the SDK resolves them on
    the first request, so a missing key surfaces from :class:`RouterUnavailableError`."""
    profile_data = _load_profile_or_exit(wiki_id)
    return AnthropicRouter(
        GraphEntityResolver(driver, wiki_id),
        comparable_properties=profile_data.retrieval.comparable_properties,
    )


def _exit_on_unavailable_router(exc: RouterUnavailableError) -> typer.Exit:
    """Turn an unusable router into a clean exit rather than a traceback."""
    console.print(f"[red]routing is unavailable[/] - {exc}")
    return typer.Exit(code=1)


def _render_context(context: RetrievedContext, show_context: bool) -> None:
    """Print the route and what it retrieved."""
    route = context.route
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("intent", f"{route.intent}{' (downgraded)' if route.downgraded else ''}")
    table.add_row("template", route.template or "-")
    table.add_row("confidence", f"{route.confidence:.2f}")
    table.add_row("entities", ", ".join(e.title for e in route.entities) or "-")
    if route.unresolved:
        table.add_row("unresolved", ", ".join(route.unresolved))
    table.add_row("context", f"{context.total_tokens:,} tokens in {len(context.blocks)} blocks")
    table.add_row("latency", f"{context.latency_ms:.0f} ms")
    console.print(table)

    for block in context.blocks:
        if block.label == "graph_facts":
            console.print(f"\n[bold]graph facts[/] ({block.tokens:,} tokens)")
            console.print(block.text)
        elif show_context:
            console.print(f"\n[bold]passages[/] ({block.tokens:,} tokens)")
            console.print(block.text)
        else:
            console.print(f"\n[bold]passages[/] {len(context.chunk_ids)} chunks (--show-context)")
            for chunk_id in context.chunk_ids:
                console.print(f"  {chunk_id}")
    console.print()


def _render_router_report(report: RouterReport, show_failures: bool) -> None:
    """Print the report the Phase 5 exit criterion is checked against."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("questions", f"{len(report.outcomes):,}")
    intent_style = "green" if report.intent_accuracy >= 0.9 else "yellow"
    template_style = "green" if report.template_accuracy >= 0.9 else "yellow"
    table.add_row("intent accuracy", f"[{intent_style}]{report.intent_accuracy:.1%}[/]")
    table.add_row("template accuracy", f"[{template_style}]{report.template_accuracy:.1%}[/]")
    table.add_row("downgraded to both", f"{report.downgraded:,}")
    table.add_row("latency", f"p50 {report.p50_ms:.0f} ms, p95 {report.p95_ms:.0f} ms")
    console.print(table)

    verdict = "[green]meets[/]" if report.meets_target else "[yellow]below[/]"
    console.print(f"\n{verdict} the Phase 5 target of 90% on both measures")

    if report.unresolved_entities:
        console.print(
            f"\n[yellow]entities that resolved to nothing[/] ({len(report.unresolved_entities)})"
        )
        for mention in report.unresolved_entities:
            console.print(f"  {mention}")

    if show_failures and report.failures:
        console.print(f"\n[bold]failures[/] ({len(report.failures)})")
        for outcome in report.failures:
            console.print(f"  {outcome.question}")
            console.print(
                f"    wanted {outcome.expected_intent}/{outcome.expected_template or '-'}, "
                f"got {outcome.route.intent}/{outcome.route.template or '-'}"
            )
    console.print()


def _render_index_report(report: IndexReport) -> None:
    """Print what one index build produced."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("provider", f"{report.provider} ({report.dimensions} dims)")
    table.add_row("articles seen", f"{report.pages_seen:,}")
    table.add_row("excluded by title", f"{report.pages_excluded:,}")
    table.add_row("articles yielding nothing", f"{report.pages_without_chunks:,}")
    table.add_row("chunks", f"{report.chunks:,}")
    table.add_row("mean chunk size", f"~{report.mean_tokens:.0f} tokens")
    table.add_row("chunks naming an entity", f"{report.mention_coverage:.1%}")
    console.print(table)
    console.print()


def _render_recall(report: RecallReport, show_misses: bool) -> None:
    """Print the recall report the Phase 4 exit criterion is checked against."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("provider", report.provider)
    table.add_row("queries scored", f"{len(report.scored):,}")
    for k, value in report.recalls.items():
        style = "green" if k == 10 and report.meets_target else "default"
        table.add_row(f"recall@{k}", f"[{style}]{value:.1%}[/]")
    table.add_row("latency", f"p50 {report.p50_ms:.0f} ms, p95 {report.p95_ms:.0f} ms")
    console.print(table)

    verdict = "[green]meets[/]" if report.meets_target else "[yellow]below[/]"
    console.print(f"\n{verdict} the Phase 4 target of 90% recall@10")

    if report.unlabeled:
        console.print(
            f"\n[yellow]{len(report.unlabeled)} queries not scored[/] "
            "(no labeled page is in the index - fix the label or the crawl)"
        )
        for outcome in report.unlabeled:
            console.print(f"  {outcome.query}  ->  {list(outcome.relevant)}")

    if show_misses and report.misses:
        console.print(f"\n[bold]misses[/] ({len(report.misses)})")
        for outcome in report.misses:
            console.print(f"  {outcome.query}")
            console.print(f"    wanted:    {list(outcome.relevant)}")
            console.print(f"    retrieved: {list(dict.fromkeys(outcome.retrieved))}")
    console.print()


def _render_sweep(reports: list[RecallReport]) -> None:
    """Print recall against latency for each ``ef_search``, which is how one is chosen."""
    table = Table(box=None, pad_edge=False)
    table.add_column("ef_search", justify="right")
    table.add_column("recall@10", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    for report in reports:
        style = "green" if report.meets_target else "yellow"
        table.add_row(
            str(report.ef_search),
            f"[{style}]{report.recall_at(10):.1%}[/]",
            f"{report.p50_ms:.0f}",
            f"{report.p95_ms:.0f}",
        )
    console.print(table)
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
