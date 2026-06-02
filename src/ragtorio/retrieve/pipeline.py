"""Question in, labeled context out: the whole of Phase 5 in one object.

Kept apart from the CLI so that Phase 6's API handler and the benchmark's two
baselines can all drive the same path. The baselines are the reason ``intent`` can be
forced: a vector-only run over identical chunks and a graph-only run over the identical
graph is how the benchmark shows whether the hybrid is actually worth anything, and
that comparison is worthless if the three configurations differ anywhere but here.
"""

from __future__ import annotations

import time

from ragtorio.retrieve.graph import GraphRetriever
from ragtorio.retrieve.log import RoutingLog
from ragtorio.retrieve.merge import DEFAULT_TOKEN_BUDGET, merge
from ragtorio.retrieve.models import Intent, RetrievedContext, Route
from ragtorio.retrieve.router import Router
from ragtorio.retrieve.vector import VectorRetriever


class RetrievalPipeline:
    """Routes a question, retrieves both halves as the route asks, and merges them."""

    def __init__(
        self,
        wiki: str,
        router: Router,
        graph: GraphRetriever | None = None,
        vector: VectorRetriever | None = None,
        log: RoutingLog | None = None,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> None:
        self._wiki = wiki
        self._router = router
        self._graph = graph
        self._vector = vector
        self._log = log
        self._token_budget = token_budget

    def retrieve(self, question: str, force_intent: Intent | None = None) -> RetrievedContext:
        """Everything one question retrieves.

        ``force_intent`` skips nothing but the decision: the router still runs, because
        its entity resolution is what the graph templates and the vector filter both
        need, and a baseline that also lost entity resolution would be measuring two
        changes at once.
        """
        started = time.perf_counter()
        route = self._router.route(question)
        if force_intent is not None:
            route = route.model_copy(update={"intent": force_intent})

        graph_result = self._graph.run(route) if self._graph and route.uses_graph else None
        passages = self._vector.run(route) if self._vector and route.uses_vector else ()

        context = merge(
            route,
            graph=graph_result,
            passages=passages,
            token_budget=self._token_budget,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        self._record(context)
        return context

    def _record(self, context: RetrievedContext) -> None:
        """Log the decision, but never fail a question over it."""
        if self._log is None:
            return
        try:
            self._log.record(self._wiki, context)
        except Exception:  # a logging failure must not lose an answer
            return


def route_only(router: Router, question: str) -> Route:
    """Just the routing decision. What ``ragtorio route eval`` measures."""
    return router.route(question)
