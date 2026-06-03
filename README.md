# Ragtorio

Ask a crafting-game wiki a question and get an answer with sources you can click.

Ragtorio reads the [Factorio wiki](https://wiki.factorio.com), pulls a knowledge graph
out of its infobox templates, indexes the rest of the prose for semantic search, and
answers questions from both. *What raw ore does one electronic circuit cost* is a walk
down the recipe graph; *why does my refinery stall* is a paragraph somebody wrote. Most
real questions are a bit of each.

**No model touches ingestion.** The graph is built by parsing templates, so no edge in it
is hallucinated - and every sentence of an answer cites a passage that was actually
retrieved, checked mechanically rather than taken on trust.

A second wiki is a YAML file, not a rewrite.

```
$ ragtorio ask factorio "what raw ore does one electronic circuit cost"

intent      graph
template    recipe_tree
entities    Electronic circuit

graph facts
Electronic circuit: 1
  Iron plate: 1
    Iron ore: 1 (raw)
  Copper cable: 3
    Copper plate: 1.5
      Copper ore: 1.5 (raw)
Raw materials for 1 Electronic circuit: Copper ore 1.5, Iron ore 1.
```

## Getting started

```bash
make install                # venv + dependencies
make install-embed          # the local embedding model (pulls PyTorch, ~2GB)
make up                     # postgres (pgvector) + neo4j via docker compose
cp .env.example .env        # then set RAGTORIO_CONTACT_EMAIL
```

Wiki operators expect a working contact address in the User-Agent, so set that before
crawling anything. Answering also needs `ANTHROPIC_API_KEY`; ingestion does not.

## Running it

Build the corpus once, top to bottom. Each step reads what the one before it wrote, and
everything after the crawl is recomputable without touching the wiki again.

```bash
ragtorio init-db                      # create the tables
ragtorio harvest factorio             # crawl pages, redirects, categories into Postgres
ragtorio extract factorio             # infobox templates -> fact rows, with a coverage report
ragtorio graph load factorio          # resolve facts into nodes and edges, load into Neo4j
ragtorio graph check factorio         # orphans, incomplete recipes, unresolved references
ragtorio index build factorio         # article prose -> embedded, citable chunks
```

Then ask it things:

```bash
ragtorio ask factorio "what research do I need before advanced oil processing"
ragtorio ask factorio "how does oil cracking work" --show-context
ragtorio ask factorio "what uses sulfuric acid" --retrieve-only    # skip the model
ragtorio serve factorio                                            # HTTP on :8000
```

Over HTTP:

```bash
curl -s localhost:8000/health
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "what raw ore does one electronic circuit cost"}'
curl -N localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "how does oil cracking work", "stream": true}'
```

`POST /ask` returns the answer, its citations, the route that produced it and a
`grounded` flag. `GET /health` queries both stores and answers 503 when either is down.
`make help` lists every target; `make check` runs ruff, mypy strict and the tests.

## How it works

```
  wiki API
     |  harvest     two passes: list revisions, fetch only what changed
     v
  Postgres raw_page ----------------------------+
     |  extract     infobox templates -> facts   |  chunk + embed
     v                                           v
  fact rows                                 chunk + pgvector
     |  resolve     ids, aliases, the              |
     v              item/recipe split              |
  Neo4j graph                                      |
      \                                           /
       \        route (haiku: intent, template, entities)
        +---------------+-------------------------+
                        v
             graph facts + passages, labeled
                        |  answer (opus)
                        v
            grounded answer + checked citations
                        |
                   CLI  |  POST /ask
```

**Harvest.** Two passes per namespace: list every page's current revision id (cheap, 500
at a time), then fetch wikitext only where that id changed, so a re-crawl of an unchanged
wiki downloads no article text at all. Translated subpages (`Iron plate/de`) are dropped
by intersecting the title shape with the language codes the wiki itself reports, so a
real page like `Blueprint/tips` survives.

**Extract.** Factorio keeps infoboxes on their own pages in a separate namespace, one per
article. Each one's `prototype-type` decides its ontology labels through the profile's
`type_map`, and mapped fields become `Fact` rows through named parsers: scalars, `A + B +
C` lists, and the recipe grammar, where a fractional output amount is read as a
probability rather than a quantity. Parsers are pure functions tested against committed
wikitext, so the whole extraction layer runs offline in CI.

**Graph.** Facts resolve into four labels (`Item`, `Recipe`, `Station`, `Unlock`) and six
relationship types, written with `MERGE` so a reload after a re-extraction is always safe.
A page that is both an item and carries its own recipe becomes two nodes, because crafting
time belongs to the recipe and stack size belongs to the item. A reference that resolves
to nothing is logged, never invented. Raw materials are not a label - an item with no
inbound `PRODUCES` edge is raw, which is how ore and crude oil fall out for free.

**Index.** Articles are split on their own `== headings ==`, never mid-sentence, and each
chunk keeps its heading trail so a passage can be labeled *Oil processing > Tips* rather
than arriving context-free. Recipe tables are dropped from the text - that is the graph's
territory - but the entities they name are kept, resolved through the same aliases the
graph uses, so a search can be narrowed to the entities a question mentions. Embeddings
are local by default (`BAAI/bge-base-en-v1.5`) so a result is reproducible.

**Route.** One cheap model call (`claude-haiku-4-5`) returns an intent, one query template
from a closed enum, and the entities it thinks the question names - all three enforced by
structured outputs. **The model never writes a query.** The templates are `.cypher` files
and everything variable arrives as a bound parameter. Every entity is resolved against the
graph before anything is bound, so an invented one fails visibly instead of matching
nothing and reading as "the wiki does not say". Low confidence widens to retrieving both
halves rather than failing.

| Template | Question shape |
|---|---|
| `recipe_tree` | what a thing is made of, down to raw materials, amounts multiplied per level |
| `unlock_chain` | the research that gates something, and its prerequisite chain |
| `consumers_of` | what uses a given item, with amounts and the machines that do it |
| `tier_compare` | things sharing a wiki category, ordered by a numeric property |

**Answer.** `claude-opus-5` gets the two kinds of evidence in separate labeled sections
and is told what each means: a graph fact is an infobox parameter extracted mechanically,
a passage is prose that may describe an older version of the game. Recipe trees arrive as
nested lists with a running total; research chains arrive numbered, because the order is
the answer.

Then every citation is checked against the chunk ids that question actually retrieved. An
unknown id triggers exactly one regeneration with the offending *sentences* quoted back -
a retry that only says "a citation was wrong" gets the same text with the citations
quietly deleted, which is worse. If the second attempt fails too, those claims ship marked
`unverified` rather than silently cleaned up. Citations render as
`index.php?title=X&oldid=N`, pinned to the revision the chunk was read at, because an
answer linking to the live page cannot be checked six months later.

## Built with

| | |
|---|---|
| Python 3.12, typer, pydantic | CLI, config, and a validated model at every boundary |
| Postgres 16 + pgvector | raw crawl, fact rows, chunks with an HNSW index |
| Neo4j 5 | the resolved graph: four labels, six relationship types |
| mwparserfromhell | template parsing |
| sentence-transformers (`bge-base-en-v1.5`) | local embeddings; Voyage as the alternative |
| Anthropic API | `claude-haiku-4-5` routing, `claude-opus-5` answering |
| FastAPI + uvicorn | `POST /ask`, `GET /health` |
| ruff, mypy strict, pytest | 470 tests, 80% coverage gate, green in CI |

## Adding a wiki

```bash
ragtorio probe https://example.wiki/api.php     # what does it run, and is its data real?
cp wikis/factorio.yaml wikis/newwiki.yaml       # edit namespaces, fields, type map
ragtorio profile newwiki                        # fails loudly on a typo
```

One [YAML file](wikis/factorio.yaml) carries everything wiki-specific: the API and rate
limit, which namespaces to crawl and how translations are marked, whether infoboxes live
on their own pages or inline in articles, which fields become which facts through which
parser, how a page's type maps onto the ontology, and how prose is chunked. New infobox
grammar means one pure function in `extract/parsers/`, named in the profile; a profile
naming a parser that does not exist fails when it loads.

[`tests/unit/test_second_wiki.py`](tests/unit/test_second_wiki.py) is the guard: a profile
sharing no vocabulary with Factorio, driven through config, extraction and resolution. It
is what caught five things that had quietly hardcoded themselves. The ontology itself
stays fixed - a wiki about films would not fit, and should not.

`ragtorio probe` is also worth running before committing to a wiki at all. Factorio's
advertises Semantic MediaWiki, which would hand you a typed graph for free; it is
installed and **empty**, which is why everything here parses templates instead.

## Honest status

Measured against a full crawl on 2026-09-13, written up under
[`docs/coverage/`](docs/coverage/):

| | |
|---|---|
| English articles crawled | 1,179 |
| Infobox pages | 590 |
| Fact rows | 7,181 |
| Graph | 867 nodes, 2,921 edges |
| Chunks | 2,301, 43% of them naming a graph entity |

**Not measured yet.** Retrieval recall, router accuracy and answer quality all need
credentials this machine does not have, and the benchmark - accuracy by hop count for
hybrid against vector-only and graph-only, with questions generated from the game's own
data dump - is not built. When it lands, its numbers belong at the top of this file.

**Known hard cases.** Ratio questions ("how many furnaces feed one assembler") need
arithmetic over rates the graph stores but does not relate. Questions about a specific
game version get answered from whatever the wiki currently says. The per-IP rate limit on
`/ask` lives in the process, so it is right for one container and wrong behind a load
balancer.

## Licensing

Wiki content is **CC BY-NC-SA 3.0** (Factorio Wiki, Wube Software). The non-commercial
clause is real: this is a research and portfolio project, not the basis for a paid
product. Each wiki profile carries its own license and attribution string, because they
differ - Fandom is CC BY-SA with no NC clause.

The code in this repository is MIT.
