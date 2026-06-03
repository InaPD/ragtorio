# Phase 6 - grounded answering, and what a citation is worth

Written 2026-09-14, against the same loaded Factorio graph (867 nodes, 2,921 edges) and
chunk index (2,301 chunks) as the Phase 4 and Phase 5 numbers.

The phase's exit criterion is twenty answered questions under `docs/examples/`, 100% of
citations resolving to retrieved chunks, and a p95 latency written down. **That run has
not happened on this machine**: it needs an Anthropic key and the local embedding model,
neither of which is installed here, and the same gap already blocks the Phase 4 recall
number and the Phase 5 router accuracy. What exists instead is the mechanism that
produces all three numbers - `ragtorio examples run factorio` writes the documents and
computes the rate and the percentiles from the answers it just generated, so the figure
in the document and the figure the system produced cannot diverge. The question set is
[`wikis/factorio.examples.yaml`](../../wikis/factorio.examples.yaml), and it is chosen
to expose failure rather than to demo well: six multi-hop quantity questions, three
research chains, three fan-outs, one numeric comparison, five prose questions with no
graph fact to fall back on, and two the wiki cannot answer at all.

Everything below **is** measured, by tests that run in CI with no key.

## The citation check

The claim "every citation resolves to something retrieved" is only worth making if the
system can be shown to catch the cases where it would not. The validator is tested
against each of them:

| Case | What happens |
|---|---|
| `[factorio:1:0000]`, retrieved | Resolves to `Electronic circuit`, revision 194123 |
| `[factorio:1:0001]`, one digit off | Unknown; the sentence is quoted back |
| `[graph]` with graph facts present | Recorded as a graph citation |
| `[graph]` with none retrieved | Unknown. Citing a source that was not consulted is fabrication whether or not the id looks like an id |
| `[as of 1.1]` | Not a citation. Bracketed prose is left alone |
| `[a, b]` in one bracket | Both checked |
| The same id twice | One citation |

A failure produces the *sentences*, not just the ids. The first version of the
regeneration prompt said only that a citation was wrong, and the obvious model response
to that is to return the same text with the citations deleted - the same claims, now
unfalsifiable. Naming the claims is what makes the retry fix them.

## One retry, then the truth

Attempt one generates. If a citation resolves to nothing, attempt two gets the note and
the offending claims. If attempt two fails the same way, the answer ships with those
claims listed as `unverified`, on the object, in the API response, and in the written
document. Not a third attempt: a model that has been shown exactly which sentence was
wrong and cited a phantom source again is not going to be talked round.

Streamed answers get one attempt only. The reader has already seen the first one, and
swapping it for a second mid-response is worse than saying which claim is unsourced; the
final SSE event carries that instead.

## What the model is actually sent

Two labeled sections, never concatenated, plus a citation instruction built per question
from what was retrieved - a model told it may write `[graph]` on a question that
retrieved no graph facts is being invited to fabricate one.

A recipe tree reaches the prompt as a nested list rather than the retriever's flat lines,
with a raw-material total at the end:

```
- 1 Electronic circuit
  - 1 Iron plate
    - 1 Iron ore (raw material)
  - 3 Copper cable
    - 1.5 Copper ore (raw material)

Raw material total for 1 Electronic circuit: 1.5 Copper ore, 1 Iron ore.
```

A branch cut for looping says so ("the chain loops here") rather than rendering as a raw
material - the Phase 5 sulfuric-acid cycle, arriving in a prompt, would otherwise tell
the model that sulfuric acid is mined. A research chain is numbered, and a technology
that appears twice through a diamond in the prerequisite graph appears once.

## The API

`POST /ask` and `GET /health`, over the same pipeline the CLI drives, with the
integration tests running the whole path against real stores and a stubbed model.

- Question length is bounded (3-500 characters) and unknown fields are rejected. The
  question goes into a prompt; an unbounded one is a funding model for whoever finds it.
- Per-IP sliding window, in process, `Retry-After` on refusal. Correct for one
  container and wrong for two, which is the README's note rather than a Redis
  dependency.
- `/health` runs a real query against Postgres and Neo4j and answers 503 when either is
  down. A check that only proves the web server started is how a deployment with an
  unreachable graph looks healthy for a week.
- No stack trace reaches a response body, and the "model unavailable" path is separated
  from the generic one because it is the single internal failure a caller can act on.
  The test asserts that a psycopg error's text - which names the table and the user -
  does not appear in the 500.

## Two Phase 5 shapes had to change

`RetrievedContext` carried `chunk_ids` and dropped the chunks. A citation has to render
as `index.php?title=X&oldid=N`, and the title and revision live on the chunk, so it now
carries the passages whole and derives the ids. Second, `retrieve/graph.py` was building
`GraphRetriever` in the CLI without the profile's `recipe_suffix` or its
`retrieval` section, so `tier_compare` silently fell back to a default property in the
one code path a user actually runs. Both are fixed.

## Still unmeasured

The twenty answers, their citation resolution rate, and p95 latency. `make examples`
with a key and `make install-embed` produces them, and the document it writes is the
artifact the exit criterion asks for.
