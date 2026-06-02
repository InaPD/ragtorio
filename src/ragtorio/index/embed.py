"""Turning passages into vectors, behind one interface.

The default is a local model - ``BAAI/bge-base-en-v1.5``, 768 dimensions - because
reproducibility matters more here than the last two points of recall. A benchmark that
reports "hybrid beats vector-only" is worth nothing if the vector-only baseline is a
hosted endpoint that was silently retrained between the two runs. Local also means the
whole index can be rebuilt after a chunker change for the cost of some CPU time, which
is what makes the chunker safe to change at all.

The Voyage provider is here to keep the seam honest rather than because the project
needs it: an interface with exactly one implementation is a guess about what varies.
Writing the second one is what surfaced the things that really do differ between
embedding providers - query and document vectors are asymmetric, dimensions are a
property of the provider and not of the schema, and batching limits are per-vendor.

**Asymmetry is not optional.** Both bge and Voyage are trained so that a question and
the passage answering it are encoded differently. Embedding a query the same way as a
document costs several points of recall, silently, and looks exactly like a chunking
problem when you go looking for it.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Sequence
from typing import Any, Protocol

import httpx

#: What bge's own model card prescribes for the query side of a retrieval pair.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_DIMENSIONS = 768

VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_DEFAULT_MODEL = "voyage-3.5"
#: Voyage's models offer 2048, 1024, 512 or 256 - never 768. Switching to Voyage
#: therefore means the ``chunk.embedding`` column changes width too, which is why
#: :func:`~ragtorio.index.postgres.ensure_embedding_dimension` exists.
VOYAGE_DEFAULT_DIMENSIONS = 1024


class EmbeddingProvider(Protocol):
    """One embedding model, as the rest of the system needs it."""

    @property
    def name(self) -> str:
        """Provider and model, for the report and for knowing what an index holds."""
        ...

    @property
    def dimensions(self) -> int:
        """Vector width. Must match the ``chunk.embedding`` column."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Vectors for passages, in the order given."""
        ...

    def embed_query(self, text: str) -> tuple[float, ...]:
        """A vector for a question, encoded for the query side of the pair."""
        ...


class HashingEmbedder:
    """A deterministic, dependency-free stand-in: hashed bag of words, L2-normalised.

    Not a semantic model and not pretending to be one. It exists so that the store,
    the search path and the recall harness can all be tested end to end without a
    2GB download, and so that ``--provider hashing`` gives a lexical-overlap baseline
    to sanity-check a suspiciously good recall number against. Same text in, same
    vector out, on any machine and any Python build.
    """

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return f"hashing/{self._dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed(text)

    def _embed(self, text: str) -> tuple[float, ...]:
        weights = [0.0] * self._dimensions
        for token in text.casefold().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            weights[int.from_bytes(digest[:4], "big") % self._dimensions] += 1.0
        return _normalize(weights)


class SentenceTransformerEmbedder:
    """The default: a local sentence-transformers model, typically bge-base-en-v1.5.

    Imported lazily so that the dependency (and the ~2GB of PyTorch behind it) is
    needed only by whoever actually builds an index. Everything else in the pipeline,
    CI included, runs without it.
    """

    def __init__(
        self, model_name: str = DEFAULT_MODEL, query_prefix: str = BGE_QUERY_PREFIX
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the install extras
            raise RuntimeError(
                "sentence-transformers is not installed. Install the embedding extra "
                '(pip install -e ".[embed]") or pass --provider hashing.'
            ) from exc
        self._model_name = model_name
        self._query_prefix = query_prefix
        self._model = SentenceTransformer(model_name)

    @property
    def name(self) -> str:
        return f"sentence-transformers/{self._model_name}"

    @property
    def dimensions(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._encode([self._query_prefix + text])[0]

    def _encode(self, texts: list[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        # normalize_embeddings makes the dot product a cosine, which is what both the
        # pgvector index and the in-memory store assume.
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [tuple(float(value) for value in vector) for vector in vectors]


class VoyageEmbedder:
    """The hosted alternative. Needs ``VOYAGE_API_KEY``.

    ``model`` and ``dimensions`` are passed together because they are not independent:
    Voyage's models accept 2048, 1024, 512 or 256, and asking for a width a model does
    not support is an API error rather than a silent truncation. The pair also has to
    match the ``chunk.embedding`` column, which is checked when an index is built.
    """

    def __init__(
        self,
        model: str = VOYAGE_DEFAULT_MODEL,
        dimensions: int = VOYAGE_DEFAULT_DIMENSIONS,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        batch_size: int = 128,
    ) -> None:
        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise RuntimeError(
                "VOYAGE_API_KEY is not set. Set it, or use the default local provider."
            )
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._client = client or httpx.Client(
            headers={"Authorization": f"Bearer {key}"}, timeout=60.0
        )

    @property
    def name(self) -> str:
        return f"voyage/{self._model}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._post(list(texts[start : start + self._batch_size]), "document"))
        return vectors

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._post([text], "query")[0]

    def _post(self, texts: list[str], input_type: str) -> list[tuple[float, ...]]:
        if not texts:
            return []
        response = self._client.post(
            VOYAGE_API_URL,
            json={
                "model": self._model,
                "input": texts,
                "input_type": input_type,
                "output_dimension": self._dimensions,
            },
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        # Voyage documents the response as ordered by input index, but says so rather
        # than guaranteeing it, and every item carries its index. Sorting costs
        # nothing and removes the question.
        items = sorted(payload["data"], key=lambda item: int(item["index"]))
        return [_normalize([float(value) for value in item["embedding"]]) for item in items]


def build_provider(
    provider: str,
    model: str | None = None,
    dimensions: int | None = None,
) -> EmbeddingProvider:
    """An :class:`EmbeddingProvider` by name, as the CLI's ``--provider`` flag gives it."""
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(model or DEFAULT_MODEL)
    if provider == "voyage":
        return VoyageEmbedder(
            model or VOYAGE_DEFAULT_MODEL, dimensions or VOYAGE_DEFAULT_DIMENSIONS
        )
    if provider == "hashing":
        return HashingEmbedder(dimensions or DEFAULT_DIMENSIONS)
    raise ValueError(
        f"unknown embedding provider {provider!r}; "
        "known providers: sentence-transformers, voyage, hashing"
    )


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity. Used by the in-memory store and by the recall harness."""
    if len(left) != len(right):
        raise ValueError(f"dimension mismatch: {len(left)} != {len(right)}")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    magnitude = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / magnitude if magnitude else 0.0


def _normalize(values: list[float]) -> tuple[float, ...]:
    """Scale to unit length, so a dot product is a cosine. An all-zero vector stays so."""
    magnitude = math.sqrt(sum(value * value for value in values))
    if not magnitude:
        return tuple(values)
    return tuple(value / magnitude for value in values)
