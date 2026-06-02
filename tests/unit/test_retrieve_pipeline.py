"""The pipeline: which halves run, and what happens when logging fails."""

from __future__ import annotations

from ragtorio.index.models import Chunk, ChunkMatch
from ragtorio.retrieve.log import InMemoryRoutingLog
from ragtorio.retrieve.models import GraphResult, RetrievedContext, Route
from ragtorio.retrieve.pipeline import RetrievalPipeline


class StubRouter:
    def __init__(self, route: Route) -> None:
        self._route = route
        self.questions: list[str] = []

    def route(self, question: str) -> Route:
        self.questions.append(question)
        return self._route.model_copy(update={"question": question})


class StubGraph:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, route: Route) -> GraphResult:
        self.calls += 1
        return GraphResult(template="recipe_tree", lines=("Iron plate: 1",))


class StubVector:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, route: Route) -> list[ChunkMatch]:
        self.calls += 1
        return [
            ChunkMatch(
                chunk=Chunk(
                    chunk_id="c1",
                    wiki="factorio",
                    page_id=1,
                    title="Oil",
                    revision_id=1,
                    text="prose about oil that is long enough to be worth keeping",
                ),
                score=0.8,
            )
        ]


class ExplodingLog:
    def record(self, wiki: str, context: RetrievedContext) -> None:
        raise RuntimeError("postgres is down")

    def count(self, wiki: str) -> int:
        return 0


def build(intent: str) -> tuple[RetrievalPipeline, StubGraph, StubVector, InMemoryRoutingLog]:
    graph, vector, log = StubGraph(), StubVector(), InMemoryRoutingLog()
    pipeline = RetrievalPipeline(
        "factorio",
        router=StubRouter(Route(question="q", intent=intent, template="recipe_tree")),  # type: ignore[arg-type]
        graph=graph,  # type: ignore[arg-type]
        vector=vector,  # type: ignore[arg-type]
        log=log,
    )
    return pipeline, graph, vector, log


def test_a_graph_route_does_not_pay_for_a_vector_search():
    pipeline, graph, vector, _ = build("graph")
    pipeline.retrieve("q")
    assert (graph.calls, vector.calls) == (1, 0)


def test_a_vector_route_does_not_query_the_graph():
    pipeline, graph, vector, _ = build("vector")
    pipeline.retrieve("q")
    assert (graph.calls, vector.calls) == (0, 1)


def test_both_runs_both():
    pipeline, graph, vector, _ = build("both")
    context = pipeline.retrieve("q")
    assert (graph.calls, vector.calls) == (1, 1)
    assert [b.label for b in context.blocks] == ["graph_facts", "passages"]


def test_forcing_an_intent_still_routes_first():
    """The baselines need the router's entity resolution; only its choice is overridden."""
    pipeline, graph, vector, _ = build("both")
    pipeline.retrieve("q", force_intent="vector")
    assert (graph.calls, vector.calls) == (0, 1)


def test_every_question_is_logged_with_what_it_retrieved():
    pipeline, _, _, log = build("both")
    pipeline.retrieve("what does a circuit cost")

    assert log.count("factorio") == 1
    _, context = log.entries[0]
    assert context.route.question == "what does a circuit cost"
    assert context.chunk_ids == ("c1",)


def test_a_logging_failure_does_not_lose_the_answer():
    pipeline = RetrievalPipeline(
        "factorio",
        router=StubRouter(Route(question="q", intent="vector")),
        vector=StubVector(),  # type: ignore[arg-type]
        log=ExplodingLog(),
    )
    assert pipeline.retrieve("q").chunk_ids == ("c1",)


def test_latency_is_measured_across_the_whole_retrieval():
    pipeline, _, _, _ = build("both")
    assert pipeline.retrieve("q").latency_ms > 0
