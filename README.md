# Ragtorio

Knowledge-graph and vector RAG over the [Factorio wiki](https://wiki.factorio.com), built
so a second crafting-game wiki is a config file rather than a rewrite.

Multi-hop questions players actually ask - *what raw ore does a rocket silo cost*, *what
research do I need before oil processing* - are answered from a knowledge graph extracted
**deterministically from wiki templates**, with mechanics prose retrieved alongside from a
vector index. No model touches the ingestion pipeline, so no edge in the graph is
hallucinated.

> **Status: Phases 0-6 of 9 complete.** Scaffolding and wiki profile, the harvester,
> template extraction, entity resolution into a Neo4j graph, the vector index over
> article prose, routing with the four query templates, and grounded answering with
> validated citations behind `POST /ask`. The benchmark is not built yet.
> Three exit criteria are unmeasured for want of credentials on this machine: Phase 4's
> recall target needs the local embedding model installed, and Phase 5's router and
> Phase 6's twenty example answers need an Anthropic key. Everything else is measured -
> [phase4-index.md](docs/coverage/phase4-index.md),
> [phase5-retrieval.md](docs/coverage/phase5-retrieval.md),
> [phase6-answering.md](docs/coverage/phase6-answering.md).
> See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Why this wiki is harder than it looks

The Factorio wiki advertises Semantic MediaWiki, which would hand you a typed graph for
free. It is installed and **empty** - four properties, all SMW's own built-ins:

```
$ ragtorio probe https://wiki.factorio.com/api.php

structured data  extension: SemanticMediaWiki
  installed_but_empty  4 properties, all SMW built-ins; parse infobox templates instead
```

Every game wiki checked tells the same story. The Stardew Valley wiki advertises Cargo and
declares zero tables; Minecraft, Terraria, OSRS and the Fandom wikis have no structured
extension at all. **Template parsing is the only path that generalises**, which is why
`ragtorio probe` reports whether a store is *populated* rather than merely installed.

## What Phase 0 measured

Numbers from the live wiki, not estimates ([full report](docs/coverage/phase0-type-enumeration.md)):

| | |
|---|---|
| Mainspace pages, all languages | 5,335 |
| Mainspace pages, English | **1,168** |
| `Infobox:` namespace (3002), English | **590** (not translated) |
| Infobox pages carrying `prototype-type` | 553 |
| Distinct `prototype-type` values | **96** |
| Archived content | namespace 3004 (`Archive:`) |

Three things the plan had wrong until the wiki was asked directly:

1. **The recipe grammar has an output side.** 72 of 284 recipe fields look like
   `Time, 60 + Uranium-235, 40 = Uranium-235, 41 + Uranium-238, 2`, and fractional
   amounts are probabilities.
2. **Technology unlocks live in `effects`, not `allows`.** `allows` is the exact inverse
   of another technology's `required-technologies`, so mapping it would double every
   prerequisite edge.
3. **21 of 590 infobox pages open with a lowercase `{{infobox`.**

## Quick start

```bash
make install          # venv + dependencies
make up               # postgres (pgvector) + neo4j via docker compose
make check            # ruff, mypy strict, pytest with an 80% coverage gate

ragtorio probe https://wiki.factorio.com/api.php   # inspect any MediaWiki wiki
ragtorio profile factorio                          # validate the shipped wiki profile

ragtorio init-db                                   # create the tables
ragtorio harvest factorio --dry-run --limit 20     # crawl into memory, write nothing
ragtorio harvest factorio                          # full crawl into Postgres
ragtorio extract factorio                          # infobox templates -> fact rows

ragtorio graph load factorio                        # resolve facts, load into Neo4j
ragtorio graph check factorio                       # orphans, gaps, self-cycles

pip install -e ".[embed]"                          # the local embedding model (pulls PyTorch)
ragtorio index build factorio                      # article prose -> embedded chunks
ragtorio index recall factorio                     # recall@k on the labeled query set
ragtorio index recall factorio --ef-search 40,100,200   # pick the HNSW accuracy setting

export ANTHROPIC_API_KEY=...                       # routing and answering; ingestion needs no key
ragtorio ask factorio "what raw ore does one electronic circuit cost"
ragtorio ask factorio "why does my refinery stall" --show-context
ragtorio ask factorio "what uses sulfuric acid" --retrieve-only   # skip the answer
ragtorio route eval factorio --show-failures       # router accuracy on the labeled set

ragtorio examples run factorio                     # answer the 20 example questions
ragtorio serve factorio                            # POST /ask and GET /health on :8000
```

```bash
curl -s localhost:8000/health
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "what raw ore does one electronic circuit cost"}'
curl -N localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "how does oil cracking work", "stream": true}'
```

Copy `.env.example` to `.env` and set `RAGTORIO_CONTACT_EMAIL` before any crawl - wiki operators
expect a working contact address in the User-Agent.

## Harvesting

`ragtorio harvest` runs two passes per namespace. The first lists pages and their current
revision ids, which is cheap: 500 per request, no article text. The second fetches wikitext and
categories only for the pages whose revision id changed, 50 at a time because that is the most
the API will give. A re-crawl of an unchanged wiki therefore issues a handful of listing
requests and downloads no article text at all.

Translated subpages (`Iron plate/de`, about twenty per article) are dropped before the fetch
pass. The filter intersects the title shape with the language codes the wiki itself reports
through `siteinfo`, so a real page like `Blueprint/tips` survives while `Iron plate/de` does not.

Raw wikitext is stored verbatim. Everything downstream is recomputable from the `raw_*` tables
without touching the wiki again, which is what makes a parser change cheap.

## Extraction

`ragtorio extract` walks every page in the infobox namespace (`Infobox:X`, namespace 3002 on
Factorio), not the mainspace: the two correspond one to one by title, so that direction needs no
wikitext parsing to find the pair. `X`'s own page is fetched only by exact title, for its
`{{history}}` templates.

Each infobox's `prototype-type` (or, for the ~40 pages that carry none, a fallback rule keyed on
which fields are present) decides its labels through the profile's `type_map`. Mapped fields turn
into `Fact` rows: a plain scalar, an `A + B + C` list, or - for `recipe` and technology `cost` -
the shared "`Time, N` plus a list of amounts, optionally `= output list`" grammar, with a
fractional output amount read as a probability (uranium processing) rather than a quantity.

A run replaces every fact stored for the wiki, since a fact is always recomputable from
`raw_page`; nothing here is Factorio-specific either. `ragtorio extract factorio` prints a
coverage report - infobox pages seen, pages that yielded a fact, unknown parameters seen more
than five times, and any parser failure together with the value that broke it.

## The graph

`ragtorio graph load` resolves every fact into a node or an edge, then writes it into Neo4j
with `MERGE` (idempotent: re-running after a re-extraction is always safe). A few things are
not directly stated by the ontology and worth knowing:

- A page that is both an `Item` and carries its own recipe becomes **two** nodes - the item,
  and `{title} (recipe)` - since a recipe's own properties (crafting time, what it consumes)
  are not properties of the item it produces. A page already classified `Recipe` needs no split.
- **`:Station` is not a `type_map` label.** It is added, after every edge resolves, to whatever
  node is on the receiving end of a `CRAFTED_AT` edge - so the profile never hand-lists which
  machines happen to craft things.
- An item's `consumers` field carries no amount, unlike its own `recipe` field's ingredient
  list. Both describe the same edge from opposite ends of the wiki, so they are merged, and the
  version carrying a real amount wins.
- A reference that does not resolve to a real node is **logged, not invented** -
  `ragtorio graph check` lists these alongside orphans, recipes missing inputs or outputs, and
  self-cycles (a recipe that consumes what it also produces - Kovarex enrichment process is a
  legitimate one).

`recipe_tree` walks an item down to raw materials, multiplying amounts level by level in
Python. A fractional output amount (uranium processing) is read as a probability, and the
batches needed are an expected-value calculation - the same reasoning the wiki's own article
uses. It stops at an item already on the current branch, because Factorio's fluids form
real cycles and multiplying through one produces a number in the millions.

## The vector index

The graph answers questions whose shape is a join. It cannot answer *why does my oil
setup back up*, because that answer is a paragraph somebody wrote and was never a
template parameter. `ragtorio index build` turns article prose into embedded, citable
chunks so both halves can be retrieved for the same question.

- **Sections, not sliding windows.** A wiki article is already segmented by its author
  into units that answer one question each, and a `== heading ==` never cuts a
  sentence. Sections over the token budget are packed greedily from whole paragraphs.
  Every chunk keeps its heading trail, so a passage can be labelled *Oil processing >
  Setting up oil processing > Tips* rather than arriving context-free.
- **Tables are dropped from the text but not from the mentions.** A recipe table is the
  graph's territory - Phase 2 already extracted it in a form that survives being
  queried, and flattened into prose it would only dilute the section's embedding. Its
  `{{Icon}}` calls still say, correctly, what the section is about.
- **Mentions come from templates as much as from links.** This wiki writes
  `{{Icon|Crude oil|100}}` far more often than it links crude oil. Each mention is
  resolved through the same redirects and aliases the graph uses and filtered against
  the titles that actually produced facts, so `chunk.mentioned_entity_ids` joins to
  node ids instead of being a second, subtly different vocabulary. That is what lets
  Phase 5 restrict retrieval to the entities its router resolved.
- **Archived content is not indexed.** `Archive:` lives in namespace 3004 and the
  profile points the chunker at namespace 0, so serving a removed item's article as
  current cannot happen by accident.
- **Embeddings are local by default** - `BAAI/bge-base-en-v1.5`, 768 dimensions -
  because a benchmark claiming "hybrid beats vector-only" is worth nothing if the
  baseline is a hosted endpoint that was retrained between the two runs. Voyage is
  implemented as the alternative; it emits 1024 dimensions, and `index build` resizes
  the column rather than making a provider switch a hand-written migration.

`ragtorio index recall` scores the index against
[30 hand-labeled queries](wikis/factorio.recall.yaml) - deliberately the questions the
graph *cannot* answer. Labels are page titles rather than chunk ids, so they do not
have to be rewritten every time the token budget changes, and a label naming a page
that is not in the index is reported as a labelling bug rather than quietly counted as
a retrieval failure. `--ef-search 40,100,200` sweeps the HNSW accuracy setting, which
is how a value gets chosen for this index instead of copied from a blog post.

Over the real crawl this comes to 2,301 chunks from 1,179 articles, 43% of them naming
a graph entity. The first build came out at 3,951 - the `Version history/*` archive, 23
machine-generated changelog pages, was 42% of the index on its own, and the build
report's mention-coverage number is what surfaced it. Full write-up:
[docs/coverage/phase4-index.md](docs/coverage/phase4-index.md).

> The integration tests use their own Postgres database (`ragtorio_test`), so a crawl
> survives `make test`. Neo4j has no equivalent - the community edition is
> single-database - so the suite still clears the loaded graph. Re-run
> `ragtorio graph load` after it.

## Routing and retrieval

A question does not announce whether its answer is a join over template data or a
paragraph somebody wrote, so `ragtorio ask` asks a small model - `claude-haiku-4-5`, one
call, a few hundred tokens - which halves of the system to consult.

**The model never writes a query.** It returns an intent, one template name from a
closed enum, and the entity names it thinks the question mentions, all three enforced by
structured outputs. The four templates are `.cypher` files; everything variable arrives
as a bound parameter. A model that answers `DROP DATABASE` to the template field gets a
schema error, not a query.

**Every entity is resolved against the graph before any parameter is bound** - exact
title, then the aliases Phase 3 built from redirects and community shorthand, then a
full-text search whose terms are all required. A hallucinated entity fails visibly, as
an unresolved mention, rather than matching nothing inside a query and returning an
empty result that reads as "the wiki does not say".

**Low confidence widens to `both`** rather than failing. A wrong template answers a
different question convincingly; one extra query costs a few hundred tokens. So does a
graph route whose entities all failed to resolve, or one the model gave no template for.

The four templates, and what the graph had to learn to answer them:

| Template | Question shape |
|---|---|
| `recipe_tree` | what a thing is made of, down to raw materials, amounts multiplied level by level |
| `unlock_chain` | the research that gates an item, a recipe or a technology, and its prerequisite chain |
| `consumers_of` | what uses a given item, with amounts and the machines that do it |
| `tier_compare` | things sharing a wiki category, ordered by a numeric property |

Retrieved context comes back as **labeled blocks**, not one wall of text: a graph fact
was a template parameter somebody typed into an infobox, a passage is prose that may be
stale, and Phase 6's prompt can say which is which. When the token budget binds,
passages are dropped from the bottom and graph facts never are - a truncated recipe tree
is a wrong answer, one fewer passage is not. Every question, its route and the chunk ids
it retrieved land in `routing_log`.

Running the templates against real data found three things the code alone did not:
`recipe_tree` crashed on ingredients with no stated amount and exploded to twenty
million sulfuric acid on Factorio's fluid cycles, and `unlock_chain` had exactly **one**
edge to walk, because the field that reads like a prerequisite list contains science
packs and the one Phase 0 dismissed contains the technology tree. All three are fixed
and written up in [phase5-retrieval.md](docs/coverage/phase5-retrieval.md), with the
twelve template checks and the router's still-unmeasured accuracy.

## Grounded answering

`ragtorio ask` routes, retrieves, and then hands the evidence to `claude-opus-5` with
adaptive thinking on and a system prompt that is the same bytes on every request, so it
sits in front of the cache breakpoint and everything per-question comes after it.

**The two kinds of evidence stay labeled all the way into the prompt.** A recipe tree
arrives as a nested list with a quantity on every line and a raw-material total at the
end; a research chain arrives numbered, because the order is the answer. The prompt says
what each label means - a graph fact is an infobox parameter extracted mechanically, a
passage is prose that may describe a different version of the game - and what to do when
they disagree.

**Every citation is checked against the chunk ids that question actually retrieved.**
Graph facts cite `[graph]`, and citing that when no graph facts were retrieved counts as
fabricated too. An unknown id triggers exactly one regeneration, with the offending
*sentences* quoted back - a retry that only says "a citation was wrong" gets the same
text with the citations quietly removed, which is worse. If the second attempt fails,
the claim ships labeled `unverified` rather than silently cleaned up, and the API says
so in a field rather than in English.

Citations render as `index.php?title=X&oldid=N`. Pinned to the revision the chunk was
embedded from, because an answer that links to the live page cannot be checked six
months later: the page has moved on, and there is no way to tell a wrong answer from a
changed game.

A streamed answer is never regenerated. The reader has already seen the first attempt,
and replacing it mid-response would be worse than admitting the citation was bad, so the
final SSE event carries the unverified claims instead.

`POST /ask` takes `{question, stream?}` and returns the answer, its citations, the route
that produced it and a `grounded` flag; `GET /health` runs a trivial query against both
stores and answers 503 when either is down, because a health check that only proves the
web server started is how a deployment with an unreachable graph looks healthy for a
week. Requests are capped per client IP, in process - which is honest about what it is:
enough for the single container this ships as, and wrong the moment there are two of
them behind a load balancer. No stack trace ever reaches a response body.

`ragtorio examples run factorio` answers
[the twenty questions](wikis/factorio.examples.yaml) the phase is measured on and writes
them to `docs/examples/`, with the citation resolution rate and the p95 latency computed
from that run rather than typed in afterwards.

## Adding a wiki

A wiki is described by one YAML file in [`wikis/`](wikis/), validated on load.

```bash
ragtorio probe https://example.wiki/api.php     # what does it run, and is the data real?
cp wikis/factorio.yaml wikis/newwiki.yaml       # edit namespaces, fields, type map
ragtorio profile newwiki                        # fails loudly on a typo
```

The profile carries everything that differs between wikis: the API and rate limit, which
namespaces to crawl and how translations are marked, whether the infobox is a page of its
own or a template **inline in the article**, which fields become which facts and through
which parser, how a page's type maps onto the ontology's labels, what suffix the split
recipe node takes, which titles are raw materials whatever the graph says, which
properties `tier_compare` may order by, and how article prose is chunked. If the new
wiki's infobox grammar differs, add one pure function to `extract/parsers/` and name it
in the profile; a profile naming a parser that does not exist fails when it loads.

**This claim went five phases untested**, because `wikis/` held exactly one profile, and
five things turned out to be false. `location: inline` - the layout most wikis use, and
the one Factorio does *not* - was accepted by the schema and rejected by the extractor.
The `recipe` target ran Factorio's parser whatever the profile named. Four of seven
declared parser names had no implementation, so they read as an extension point and
behaved as a list of good intentions. The `(recipe)` node suffix was a literal in two
modules. And the properties a tier comparison may order by were a constant naming
Factorio's three fields, in the middle of the retrieval layer.

All five are fixed, and
[`tests/unit/test_second_wiki.py`](tests/unit/test_second_wiki.py) is the guard: a
profile that shares no vocabulary with Factorio - inline infoboxes in namespace 100, a
`Thingbox` template, `durability` and `weight`, a `[formula]` recipe suffix - driven
through config, extraction and entity resolution. The ontology itself stays fixed
(`Item`/`Fluid`/`Recipe`/`Station`/`Unlock`, six relationship types, four templates);
that is the project's scope, not an oversight. A wiki about films would not fit and
should not.

What is still untested is a *real* second wiki. A synthetic profile proves the seam
exists; it does not prove any actual wiki fits through it.

## Layout

```
wikis/factorio.yaml          the whole Factorio-specific surface
src/ragtorio/
  cli.py                     probe | profile | init-db | harvest | extract | graph |
                             index | route | ask | examples | serve
  config.py                  profile schema and loader, validated with pydantic
  db/schema.sql              raw_page, raw_redirect, raw_category, crawl_run, fact, chunk, routing_log
  db/neo4j.py                driver + schema application, mirrors db/connect.py
  harvest/client.py          rate-limited, retrying MediaWiki client
  harvest/probe.py           installed-versus-populated structured-data check
  harvest/crawl.py           two-pass crawl: list revisions, fetch only what changed
  harvest/language.py        translated-subpage filter, driven by the wiki's own codes
  harvest/store.py           the persistence a crawl needs, plus an in-memory one
  harvest/postgres.py        the Postgres implementation of that protocol
  extract/template.py        infobox pages + their article's {{history}} -> Fact rows
  extract/parsers/           factorio_recipe_expr, plus_list: pure functions, fixture-tested
  extract/repository.py      read-only access to raw_page, real and in-memory
  extract/store.py           where facts go: a run replaces a wiki's facts wholesale
  ontology/resolve.py        facts -> nodes and edges: ids, aliases, the Item/Recipe split
  ontology/schema.cypher     per-label uniqueness constraints, title and alias indexes
  ontology/load.py           batched, idempotent MERGE writes via APOC
  ontology/check.py          orphans, incomplete recipes, self-cycles
  ontology/recipe_tree.py    an item's ingredients, recursively, down to raw materials
  ontology/canonical.py      titles -> canonical titles, shared by resolution and indexing
  index/chunk.py             articles -> section chunks, with the entities they mention
  index/embed.py             EmbeddingProvider: local bge by default, Voyage, and a
                             deterministic stand-in for tests
  index/build.py             one build: chunk, embed, report what it produced
  index/store.py             where chunks go, plus an exact-search in-memory one
  index/postgres.py          pgvector + HNSW, and the column resize a provider swap needs
  index/recall.py            recall@k on the labeled set, and the ef_search sweep
  retrieve/router.py         one haiku call: intent, template, entities, all schema-bound
  retrieve/entities.py       mention -> node id, by title, alias, then full-text search
  retrieve/cypher/           the query templates, as files, parameters only
  retrieve/graph.py          runs one named template and renders its rows
  retrieve/vector.py         top-k passages, narrowed by the route's own entities
  retrieve/merge.py          labeled context blocks under a token budget
  retrieve/pipeline.py       question -> route -> both halves -> merged context
  retrieve/evaluate.py       router accuracy on the labeled question set
  answer/render.py           context -> prompt: nested recipe trees, numbered chains
  answer/generate.py         one opus call: streamed, cached prefix, refusal-aware
  answer/validate.py         citations checked against the retrieved chunk ids
  answer/pipeline.py         generate, validate, regenerate once, then admit what failed
  answer/examples.py         the 20-question run and the documents it writes
  api/app.py                 POST /ask (JSON or SSE), GET /health, one shared pipeline
  api/schemas.py             request and response bodies, validated at the boundary
  api/ratelimit.py           per-IP sliding window, in process
wikis/factorio.aliases.yaml  community shorthand ("green circuit", "blue science")
wikis/factorio.recall.yaml   30 hand-labeled queries the vector index is measured on
wikis/factorio.routing.yaml  45 hand-labeled questions the router is measured on
wikis/factorio.examples.yaml 20 questions answered into docs/examples/
tests/fixtures/wikitext/     committed wikitext, so extraction tests need no network
docs/coverage/               what the wiki actually contains
docs/examples/               answered questions, with every citation clickable
scripts/                     one-off Phase 0 measurement tools
```

## Licensing

Wiki content is **CC BY-NC-SA 3.0** (Factorio Wiki, Wube Software). The non-commercial
clause is real: this is a portfolio and research project, not a basis for a paid product.
Each wiki profile carries its own license and attribution string, because they differ -
Fandom is CC BY-SA with no NC clause.

The code in this repository is MIT.
