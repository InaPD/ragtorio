"""The FastAPI application.

Two endpoints, one code path. ``/ask`` drives the same :class:`AnswerPipeline` the CLI
does, which is the only way the benchmark numbers and the HTTP behaviour stay the same
claim about the same system.

**Resources are opened once, at startup.** A Postgres connection, a Neo4j driver and
an embedding model per request would make the first token of every answer wait on a
model load. They are built in the lifespan and shared; psycopg serialises concurrent
use of a connection internally, and the Neo4j driver is thread-safe by design.

**No stack traces leave the process.** Every unhandled exception is logged with its
traceback and answered with one sentence. A 500 that quotes a psycopg error tells a
stranger the database user, the table names and the driver version.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ragtorio.answer.generate import AnswererUnavailableError, AnthropicAnswerer
from ragtorio.answer.pipeline import AnswerPipeline
from ragtorio.answer.validate import index_url_for
from ragtorio.api.ratelimit import DEFAULT_LIMIT, RateLimiter
from ragtorio.api.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
)
from ragtorio.config import Settings, load_profile
from ragtorio.db.connect import connect
from ragtorio.db.neo4j import connect as neo4j_connect
from ragtorio.index.embed import build_provider
from ragtorio.index.postgres import PostgresChunkStore
from ragtorio.retrieve.entities import GraphEntityResolver
from ragtorio.retrieve.graph import GraphRetriever
from ragtorio.retrieve.log import PostgresRoutingLog
from ragtorio.retrieve.pipeline import RetrievalPipeline
from ragtorio.retrieve.router import AnthropicRouter, RouterUnavailableError
from ragtorio.retrieve.vector import VectorRetriever

log = logging.getLogger("ragtorio.api")

#: What a caller is told when something inside failed. Deliberately uninformative;
#: the detail is in the server's log with a traceback attached.
GENERIC_ERROR = "the request could not be completed"

#: What a caller is told when the model is unreachable. Separate from the above
#: because it is the one internal failure a caller can do something about: wait.
UNAVAILABLE_ERROR = "the answering model is unavailable; try again shortly"


@dataclass
class Health:
    """Whether each store answered a trivial query just now."""

    postgres: bool
    neo4j: bool

    @property
    def ok(self) -> bool:
        return self.postgres and self.neo4j


class Service:
    """What the endpoints need: a pipeline, a health probe, and the wiki's id."""

    def __init__(
        self,
        wiki: str,
        pipeline: AnswerPipeline,
        health: Callable[[], Health],
    ) -> None:
        self.wiki = wiki
        self.pipeline = pipeline
        self.health = health


def create_app(
    wiki_id: str = "factorio",
    service_factory: Callable[[], tuple[Service, Callable[[], None]]] | None = None,
    rate_limit: int = DEFAULT_LIMIT,
    provider: str = "sentence-transformers",
    embedding_model: str | None = None,
) -> FastAPI:
    """Build the app. ``service_factory`` is how a test supplies a fake pipeline.

    ``provider`` has to match what the index was built with: a server embedding queries
    with a different model than the chunks searches a space the chunks are not in, and
    the results come back ranked, plausible and wrong.
    """
    limiter = RateLimiter(limit=rate_limit)
    factory = service_factory or (lambda: _build_service(wiki_id, provider, embedding_model))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service, close = factory()
        app.state.service = service
        try:
            yield
        finally:
            close()

    app = FastAPI(
        title="ragtorio",
        summary="Grounded answers over a crafting-game wiki, from a graph and a vector index.",
        lifespan=lifespan,
    )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
        log.exception("unhandled error on %s", request.url.path)
        return _error(500, GENERIC_ERROR)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> Any:
        service: Service = request.app.state.service
        state = service.health()
        body = HealthResponse(
            status="ok" if state.ok else "degraded",
            wiki=service.wiki,
            postgres=state.postgres,
            neo4j=state.neo4j,
        )
        return JSONResponse(body.model_dump(), status_code=200 if state.ok else 503)

    @app.post("/ask", response_model=AskResponse)
    def ask(request: Request, body: AskRequest) -> Any:
        retry_after = limiter.check(_client_key(request))
        if retry_after is not None:
            return _error(429, "too many requests", {"Retry-After": str(int(retry_after) + 1)})

        service: Service = request.app.state.service
        if body.stream:
            return StreamingResponse(
                _stream(service.pipeline, body.question),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
            )
        try:
            answer = service.pipeline.answer(body.question)
        except (AnswererUnavailableError, RouterUnavailableError) as exc:
            log.warning("model unavailable: %s", exc)
            return _error(503, UNAVAILABLE_ERROR)
        return JSONResponse(AskResponse.of(answer).model_dump())

    return app


def _stream(pipeline: AnswerPipeline, question: str) -> Iterator[str]:
    """Server-sent events: text as it arrives, then one final event with the checks.

    The generation runs in a thread and pushes tokens through a queue because the
    pipeline is synchronous and the Anthropic stream is a context manager - inverting
    it into an async generator would mean reimplementing the SDK's stream handling.

    A streamed answer is never regenerated (see ``AnswerPipeline``), so the final
    event is where a caller learns that a claim cited nothing retrievable.
    """
    tokens: queue.Queue[str | None] = queue.Queue()
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["answer"] = pipeline.answer(question, on_text=tokens.put)
        except (AnswererUnavailableError, RouterUnavailableError) as exc:
            log.warning("model unavailable: %s", exc)
            outcome["error"] = UNAVAILABLE_ERROR
        except Exception:
            log.exception("unhandled error while streaming an answer")
            outcome["error"] = GENERIC_ERROR
        finally:
            tokens.put(None)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    while (text := tokens.get()) is not None:
        yield _event("token", {"text": text})
    worker.join()

    if "error" in outcome:
        yield _event("error", {"error": outcome["error"]})
        return
    yield _event("final", AskResponse.of(outcome["answer"]).model_dump())


def _event(name: str, payload: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def _client_key(request: Request) -> str:
    """The client's address. Behind a proxy this is the proxy, which is why the
    README says to put the real limit at the proxy if there is one - a forwarded
    header is a client-supplied string and trusting it here would let anyone reset
    their own counter by making one up."""
    return request.client.host if request.client else "unknown"


def _error(status: int, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(error=message).model_dump(), status_code=status, headers=headers
    )


def _build_service(
    wiki_id: str,
    provider: str = "sentence-transformers",
    embedding_model: str | None = None,
) -> tuple[Service, Callable[[], None]]:
    """Open everything an answer needs, once, and hand back a way to close it."""
    settings = Settings()
    profile = load_profile(wiki_id)
    conn = connect(settings.postgres_dsn)
    driver = neo4j_connect(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    embedder = build_provider(provider, embedding_model)

    retrieval = RetrievalPipeline(
        wiki_id,
        router=AnthropicRouter(
            GraphEntityResolver(driver, wiki_id),
            comparable_properties=profile.retrieval.comparable_properties,
        ),
        graph=GraphRetriever(
            driver,
            wiki_id,
            raw_items=profile.resolution.raw_item_set,
            recipe_suffix=profile.resolution.recipe_suffix,
            retrieval=profile.retrieval,
        ),
        vector=VectorRetriever(PostgresChunkStore(conn), embedder, wiki_id),
        log=PostgresRoutingLog(conn),
    )
    service = Service(
        wiki=wiki_id,
        pipeline=AnswerPipeline(retrieval, AnthropicAnswerer(), index_url_for(profile.wiki.api)),
        health=lambda: _probe(conn, driver),
    )

    def close() -> None:
        driver.close()
        conn.close()

    return service, close


def _probe(conn: Any, driver: Any) -> Health:
    """One trivial query against each store. Failures are a health state, not an error."""
    return Health(postgres=_ok(lambda: _postgres_ok(conn)), neo4j=_ok(lambda: _neo4j_ok(driver)))


def _ok(probe: Callable[[], None]) -> bool:
    try:
        probe()
    except Exception:  # any failure to answer is the answer
        return False
    return True


def _postgres_ok(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT 1")


def _neo4j_ok(driver: Any) -> None:
    with driver.session() as session:
        session.run("RETURN 1").consume()
