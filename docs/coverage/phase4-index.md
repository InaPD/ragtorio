# Phase 4 - what the vector index actually contains

Measured against a full crawl of wiki.factorio.com on 2026-09-13, the same crawl the
Phase 2 and Phase 3 numbers come from.

## The corpus

| | |
|---|---|
| English mainspace articles crawled | 1,179 |
| Excluded by title (`Version history/*`) | 23 |
| Articles yielding no chunk | 75 |
| Chunks | **2,301** |
| Mean chunk size | ~149 tokens |
| Chunks naming at least one graph entity | 43.0% |

Three of those numbers were worth chasing down.

**23 pages were producing 42% of the index.** The first build over the real crawl came
out at 3,951 chunks with only 25% of them naming any entity. The `Version history/*`
archive - machine-generated release notes, no links, no answer to any question a player
asks - accounted for 1,650 of them. It is excluded in the profile now
(`index.exclude_title_prefixes`), which is what took mention coverage from 25% to 43%
and the index from 3,951 chunks to 2,301. This is exactly what `chunks naming an
entity` is in the build report for; without it the index would have been 70% larger and
worse, and nothing would have said so.

**75 articles yield nothing**, out of 1,179. Spot-checked: disambiguation pages,
navigation stubs, and articles that are a single infobox transclusion with no prose.
That is the right answer for all three - none of them contains a passage worth
retrieving - and it is small enough not to be hiding a class of article being eaten.

**43% mention coverage is the ceiling, not a shortfall.** The other 57% is prose about
mechanics rather than about a specific item: *Balancer mechanics*, *Energy and work*,
*Glossary*. Those chunks are reachable by embedding, just not by the entity filter, and
that is the division of labour the two halves are meant to have.

## Recall

The labeled set is [30 hand-written queries](../../wikis/factorio.recall.yaml),
deliberately the questions the graph cannot answer. **All 30 scored** - every labeled
page is in the index, so none of the numbers below rest on a page the crawl missed.

Measured with the deterministic `hashing` provider, a bag-of-words stand-in:

| k | recall@k |
|---|---|
| 1 | 20.0% |
| 3 | 43.3% |
| 5 | 46.7% |
| 10 | **53.3%** |

p50 1 ms, p95 2 ms at the pgvector default `ef_search`.

**This is a baseline, not the phase's exit criterion.** The criterion is recall@10 ≥ 0.9
with `BAAI/bge-base-en-v1.5`, and 53% is roughly what lexical overlap should score
against questions that are deliberately paraphrased away from their answers' wording -
"how does food spoiling work" has to reach a page titled *Spoilage*, and nothing in the
query's bag of words says so. Running the real measurement needs the embedding extra:

```bash
pip install -e ".[embed]"     # sentence-transformers, ~2GB with PyTorch
ragtorio index build factorio # ~2,300 chunks through bge-base on CPU
ragtorio index recall factorio --show-misses
```

The hashing number is kept here anyway, because a model that cannot beat bag-of-words
by a wide margin on this set would be worth knowing about.

## ef_search

| ef_search | recall@10 | p50 | p95 |
|---|---|---|---|
| 40 | 53.3% | 1 ms | 2 ms |
| 100 | 53.3% | 1 ms | 2 ms |
| 200 | 56.7% | 11 ms | 12 ms |
| 400 | 56.7% | 11 ms | 12 ms |

The shape is the one HNSW always has - flat, then a step, then flat - and on a 2,301-row
index it is cheap everywhere. The value to ship should be re-measured with the real
embeddings before it is written into anything, since where that step falls depends on
how the vectors are distributed, not on how many of them there are.

## A hazard worth writing down

The integration tests `TRUNCATE` their tables on the DSN in `RAGTORIO_TEST_DSN`, which
defaults to the same database `ragtorio harvest` writes to. Running `make test` after a
crawl destroys it. Point `RAGTORIO_TEST_DSN` at a separate database before running the
suite against a machine that holds a crawl worth keeping.
