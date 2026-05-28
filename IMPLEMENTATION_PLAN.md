# Ragtorio - Implementation Plan

Knowledge-graph plus vector RAG over the Factorio wiki, built so a second crafting-game wiki is a config file rather than a rewrite.

**Stack:** Python 3.12 · Postgres 16 + pgvector · Neo4j 5 · mwparserfromhell · Anthropic Python SDK (1.x) · FastAPI · Docker Compose
**Primary target:** Factorio (`wiki.factorio.com`). Other crafting-game wikis are a stretch goal, not a deliverable.
**Estimate:** about 20 working days. Milestones and stretch goals in sections 7 and 8.

---

## 0. What this document is

The build plan for Ragtorio, phase by phase, with an exit criterion per phase. Everything in section 2 was verified against the live wiki API on 2026-09-13.

The thesis: **multi-hop questions players actually ask ("what raw ore does a rocket silo cost", "what research do I need before oil processing") are answered by a knowledge graph extracted 100% deterministically from wiki templates, with prose retrieved alongside from a vector index, and the hybrid beats a vector-only baseline by a widening margin as hop count grows, verified against the game's own data.**

Two simplifications were made deliberately against an earlier draft: **no images**, and **no LLM in the ingestion pipeline**. The graph is built from templates only; the model is used for routing and answering. That is a cleaner story and roughly a third less work.

## 1. Scope and non-goals

**In scope**

- Factorio wiki, English pages, current content plus explicit archived flags.
- Graph retrieval (recipes, technologies, machines), vector retrieval (mechanics prose), merged into one cited answer.
- A benchmark against `data.raw` with accuracy reported by hop count for hybrid vs vector-only.
- A wiki profile (YAML) so the harvester and extractor are not hardcoded to Factorio.

**Out of scope for v1** (see section 8 for what would bring them back)

- Images of any kind.
- LLM-based extraction of facts from prose.
- A second wiki that is tested and benchmarked. A Stardew profile may be drafted; it is not a v1 deliverable.
- RPG and lore wikis. Different ontology entirely.
- Model-generated Cypher. All graph queries are parameterised templates.
- Incremental re-crawling. Full re-harvest is fine at 5,000 pages.
- A paid product. The wiki is CC BY-NC-SA 3.0.

## 2. Verified facts about the source

### 2.1 Semantic MediaWiki is installed and empty

The wiki advertises SMW, but `Property:` (namespace 102) contains only the four built-ins that ship with it (`Foaf:homepage`, `Foaf:knows`, `Foaf:name`, `Owl:differentFrom`), and `browsebysubject` on a real page returns zero properties. **All structured data is in wikitext templates.** This is true of every game wiki probed (Stardew's Cargo is also empty; Minecraft, Terraria, OSRS and the Fandom wikis have no structured extension at all), which is why template parsing is the primary path and the profile abstraction is cheap to keep.

### 2.2 Where the data lives

Infoboxes are separate pages in a dedicated namespace, transcluded into articles:

```
Article "Iron gear wheel":      {{:Infobox:Iron gear wheel}}
Page "Infobox:Iron gear wheel": {{Infobox
                                 |prototype-type = item
                                 |internal-name  = iron-gear-wheel
                                 |stack-size     = 100
                                 |recipe         = Time, 0.5 + Iron plate, 2
                                 |producers      = Assembling machine + Player
                                 |consumers      = Artillery turret + Inserter + Lab + ...
                                 }}
```

| Fact | Value | Why it matters |
|---|---|---|
| `Infobox:` namespace id | **3002** | A mainspace-only crawl gets prose and no data. The harvester must fetch this namespace explicitly. |
| `Archive:` namespace id | **3004** | Removed content. Mark it, do not serve it as current. |
| `recipe` grammar | `Name, amount + Name, amount` | One small parser. `Time` is a pseudo-ingredient carrying crafting time. |
| `producers` / `consumers` | `A + B + C` lists | `consumers` gives every inbound `CONSUMES` edge for free; `producers` gives `CRAFTED_AT`. |
| `internal-name` | `iron-gear-wheel` | The join key to `data.raw` for the benchmark. Store it on every node. |
| Inline templates | `{{icon\|molten iron\|10}}`, `{{history\|2.0.7\|...}}` | Item refs with amounts in tables; version events. |
| Language subpages | `Iron plate/de`, ~20 languages | Filter early or the graph is 20x too big. Expect 1,500 to 2,500 English pages. |
| Stable vs documented version | wiki documents 2.1, stable release is 2.0.x | State the version stance in the README. |
| Access | `api.php`, 1 req/s, descriptive User-Agent, honour 429 | Behind Cloudflare. |
| License | CC BY-NC-SA 3.0 | Attribution in README; non-commercial. |

## 3. Architecture

```
   wikis/factorio.yaml                                       <- profile (config)
          |
   +------v---------------------------------------------------+
   |  HARVEST   MediaWiki client + crawler                    |
   |            mainspace + Infobox: + Archive:, redirects,   |
   |            categories                                    |
   +------+---------------------------------------------------+
          | raw_page, raw_redirect, raw_category   (Postgres)
   +------v---------------------------------------------------+
   |  EXTRACT   TemplateExtractor (mwparserfromhell)          |
   |            profile-driven field -> Fact mapping          |
   +------+---------------------------------------------------+
          | fact rows with provenance
   +------v---------------------------------------------------+
   |  GRAPH     entity resolution -> Neo4j loader             |
   |            4 labels, 6 relationship types                |
   +------+-------------------------+-------------------------+
          |                         |
   +------v--------+        +-------v--------+
   | Neo4j         |        | pgvector       |
   | recipe graph  |        | section chunks |
   +------+--------+        +-------+--------+
          |                         |
   +------v-------------------------v-------------------------+
   |  RETRIEVE  router (graph | vector | both)                |
   |            4 Cypher templates, vector search, merge      |
   +------+---------------------------------------------------+
   |  ANSWER    grounded generation, citations validated      |
   |            against retrieved chunk ids, oldid links      |
   +------+---------------------------------------------------+
   |  API       FastAPI  POST /ask  GET /health               |
   +----------------------------------------------------------+
```

**Stores.** Postgres holds the raw crawl, the extracted `fact` rows, the `chunk` table with embeddings, and `routing_log`. Neo4j holds only the resolved graph. Everything downstream of `raw_*` is recomputable without refetching.

**Shared key.** `chunk.chunk_id` appears on graph edges as `source_chunk_id` and in citations. One id space, one merge step.

## 4. Ontology

Four labels, six relationship types, all populated from templates.

| Label | Factorio meaning | Comes from |
|---|---|---|
| `Item` | anything craftable or minable, fluids included (`:Item:Fluid`) | `prototype-type` in `item`, `fluid`, `tool`, `ammo`, `capsule`, `module`, ... |
| `Recipe` | one transformation: inputs, outputs, time | the `recipe` field of an Item infobox, or a standalone recipe page (`Advanced oil processing`) |
| `Station` | where a recipe runs | `producers` field; `prototype-type` in `assembling-machine`, `furnace`, ... |
| `Unlock` | a technology | `prototype-type = technology` |

```
(:Recipe)-[:CONSUMES {amount}]->(:Item)
(:Recipe)-[:PRODUCES {amount, probability}]->(:Item)
(:Recipe)-[:CRAFTED_AT]->(:Station)
(:Unlock)-[:UNLOCKS]->(:Recipe)
(:Unlock)-[:REQUIRES]->(:Unlock)
(:Unlock)-[:COSTS {amount}]->(:Item)        science packs
```

Every edge carries `source_page_id`, `source_chunk_id`, `revision_id`, `confidence` (always 1.0 in v1; the field exists so LLM-sourced edges can be added later without a schema change).

Node properties: `id` (`factorio:{canonical_title}`), `title`, `internal_name`, `aliases[]`, `is_archived`, `introduced_in`, `page_id`, `revision_id`, plus numeric infobox fields as plain properties (`stack_size`, `crafting_time`, `energy_kw`, `fuel_value_mj`, `mining_time`). Numbers live on nodes, not in a separate stats table; a tier comparison is `ORDER BY n.speed` in Cypher.

**Raw materials are not a label.** An `Item` with no inbound `PRODUCES` edge is raw. `recipe_tree` terminates there. Ores, water and crude oil fall out of this rule without a `Resource` type.

**Page-to-node mapping is explicit.** "Iron gear wheel" yields an `Item` and a `Recipe`; "Advanced oil processing" yields only a `Recipe`; "Automation (research)" yields an `Unlock`. The profile's `type_map` decides, per `prototype-type` value.

## 5. Wiki profile

One YAML file. Adding a wiki later means writing another one plus, if the infobox grammar is new, one parser function. Kept because it costs almost nothing and it is what stops the code from hardcoding Factorio.

```yaml
# wikis/factorio.yaml
wiki:
  id: factorio
  api: https://wiki.factorio.com/api.php
  license: CC BY-NC-SA 3.0
  attribution: "Factorio Wiki (wiki.factorio.com), Wube Software"
  rate_limit_rps: 1
  language_filter: { strategy: subpage_suffix, keep: en }
  archived:        { namespace_id: 3004, category: Archived }
  extra_namespaces: [3002]

infobox:
  location: separate_namespace
  namespace_id: 3002
  template: Infobox
  type_field: prototype-type
  type_map:                        # completed in Phase 0 from the real value set
    item:               [Item]
    fluid:              [Item, Fluid]
    technology:         [Unlock]
    assembling-machine: [Station]
    furnace:            [Station]
    recipe:             [Recipe]
  fields:
    internal-name:          { to: prop.internal_name }
    stack-size:             { to: prop.stack_size,    type: int }
    recipe:                 { to: recipe,             parser: factorio_recipe_expr }
    producers:              { to: rel.CRAFTED_AT,     parser: plus_list }
    consumers:              { to: rel.CONSUMED_BY,    parser: plus_list }
    required-technologies:  { to: rel.REQUIRES,       parser: plus_list }
    cost:                   { to: rel.COSTS,          parser: factorio_recipe_expr }
    allows:                 { to: rel.UNLOCKS,        parser: plus_list }

inline_templates:
  history: { to: version_event, args: [version, text] }

ground_truth:
  kind: factorio_data_raw
  path: data/ground_truth/factorio/data-raw-dump.json
```

Parsers named in the profile (`factorio_recipe_expr`, `plus_list`, `int`) live in `extract/parsers/` as pure functions with fixture tests.

---

## 6. Phases

Estimates are in working days. Parsers and the extractor are tested against committed wikitext fixtures, so the whole extraction layer runs offline in CI.

### Phase 0 - Foundations (1.5 days)

**Work.** `pyproject.toml` (ruff, mypy, pytest, 80% coverage gate), `docker-compose.yml` with `postgres:16` + pgvector and `neo4j:5`, `.env.example`, GitHub Actions for lint and tests. `ragtorio probe <api_url>` as a small CLI command (already prototyped; it produced section 2). Commit wikitext fixtures for six pages and their `Infobox:` pages: `Iron gear wheel`, `Electronic circuit`, `Advanced oil processing`, `Kovarex enrichment process`, `Automation (research)`, `Assembling machine 2`. Crawl the titles and content of namespace 3002 once and enumerate the real `prototype-type` value set; complete `type_map` so every value is mapped or explicitly ignored.

**Exit.** Green CI. `docker compose up` gives two healthy databases. Profile complete against the actual value set.

### Phase 1 - Harvester (2 days)

**Work.** `harvest/client.py`: rate limit from profile, `maxlag=5`, backoff on 429 and 5xx honouring `Retry-After`, continuation, 50-title batches, User-Agent with contact. `harvest/crawl.py`: `allpages` over namespaces 0, 3002 and 3004 with `apfilterredir=nonredirects`, language filter, `revisions` (`content|ids|timestamp`), `categories`, `allredirects`. Tables: `raw_page(wiki, page_id, ns, title, revision_id, timestamp, wikitext, fetched_at)`, `raw_redirect`, `raw_category`, `crawl_run`. Skip unchanged `revision_id` on re-run.

**Exit.** English mainspace count lands in 1,500 to 2,500. Namespace 3002 count is within 10% of that. A second run fetches nothing.

### Phase 2 - Template extraction (3 days)

**Work.** `extract/models.py`: frozen pydantic `Fact(subject, subject_labels, predicate, object, object_labels, props, provenance)`. `extract/base.py`: `StructuredExtractor` protocol (one method; it exists so a Lua or Cargo extractor can be added without touching callers). `extract/template.py`: `TemplateExtractor` that walks from an article to its `Infobox:` page, applies `type_map`, maps `fields` through named parsers, walks `{{history}}` for version events. `extract/parsers/`: `factorio_recipe_expr` (handles `Time`, decimals, multi-output with probabilities as they actually appear in the fixtures), `plus_list`, scalars. Written test-first. `ragtorio extract factorio` writes `fact` rows and prints a coverage report: infobox pages seen, pages yielding facts, unknown parameters with counts, parser failures with the offending value.

**Exit.** ≥95% of infobox pages yield at least one fact. Every unknown parameter seen more than 5 times is mapped or ignored in the profile. Parser tests at 100%, extractor ≥85%.

### Phase 3 - Entity resolution and graph (3 days)

**Work.** `ontology/resolve.py`: canonical id from title, redirects become `aliases`, `wikis/factorio.aliases.yaml` for community shorthand ("green circuit", "red belt", "blue science"), normalisation, unresolved references logged rather than created. Page-to-node rules applied: `Recipe` nodes named `{title} (recipe)` where an Item page carries a recipe. `ontology/schema.cypher`: uniqueness on `id`, indexes on `title`, `aliases`, `internal_name`. `ontology/load.py`: batched `UNWIND $rows MERGE`, idempotent on `(subject, predicate, object)`. `is_archived` from namespace 3004 and `Category:Archived`; `introduced_in` from `{{history}}`. `ragtorio graph check`: orphans, recipes missing inputs or outputs, unresolved references, cycles listed (Kovarex is a legitimate cycle). First Cypher template, `recipe_tree`, with amounts multiplied level by level in Python.

**Exit.** `ragtorio ask factorio "raw ore for one electronic circuit" --graph-only` prints the nested tree with correct iron and copper totals against the `data.raw` fixture pulled early. Zero unresolved references above a documented allowlist.

### Phase 4 - Vector index (2 days)

**Work.** `index/chunk.py`: split articles on `==` headings, keep `section_path`, subdivide sections over ~800 tokens, attach `mentioned_entity_ids` from `[[links]]` and `{{icon}}` refs resolved through aliases. `index/embed.py`: `EmbeddingProvider` interface, default `sentence-transformers` with `BAAI/bge-base-en-v1.5` (768 dims, local, free, reproducible); Voyage AI as the alternative implementation. `chunk(chunk_id, wiki, page_id, revision_id, section_path, text, embedding vector(768), mentioned_entity_ids[])` with an HNSW index. `fa recall`: recall@k on 30 hand-labeled queries, used to pick `ef_search`.

**Exit.** Recall@10 ≥ 0.9 on the labeled set.

### Phase 5 - Routing and retrieval (2.5 days)

**Work.** `retrieve/router.py`: `claude-haiku-4-5`, strict enum output `{intent: graph | vector | both, template, entities[]}`, entities resolved through aliases before any query; low confidence returns `both`. Four Cypher templates as files under `retrieve/cypher/`, parameters only:

1. `recipe_tree` - recursive `CONSUMES` to raw items, depth-bounded.
2. `unlock_chain` - `REQUIRES*` path to the `Unlock` that gates a recipe or item.
3. `consumers_of` - inbound `CONSUMES` fan-out ("what uses sulfuric acid").
4. `tier_compare` - items sharing a category, ordered by a numeric property.

`retrieve/vector.py`: top-k with optional `mentioned_entity_ids` filter. `retrieve/merge.py`: dedupe by `chunk_id`, label context blocks `graph_facts` and `passages`, cap by token budget. `routing_log(question, intent, template, entities, latency_ms, chunk_ids)`.

**Exit.** Each template answers three hand-picked questions correctly (12 checks). Router ≥90% on a 40-question labeled set.

### Phase 6 - Grounded answering and API (2 days)

**Work.** `answer/render.py`: recipe trees as nested lists with quantities, chains as ordered lists. `answer/generate.py`: `claude-opus-5`, adaptive thinking, streaming, stable system prompt as cached prefix, context blocks labeled and separate, citations by `chunk_id` per claim. `answer/validate.py`: unknown citation ids trigger one regeneration naming the offending claims; a second failure marks them unverified. Citations render as `index.php?title=X&oldid=N`. `api/`: FastAPI `POST /ask {question, stream?}` and `GET /health`; pydantic validation, per-IP rate limit, no stack traces in responses.

**Exit.** 20 questions answered and saved under `docs/examples/`, 100% of citations resolve to retrieved chunks, p95 latency written down.

### Phase 7 - Benchmark (3 days)

**Work.** `bench/ground_truth.py`: load `data-raw-dump.json` (from `factorio --dump-data` on a local install, or a community export), build the true recipe and technology graph keyed by `internal_name`. `bench/generate.py`: sample paths of known hop length and template them into questions; strata of 60 to 100: 1-hop (stack size), 2-hop (ingredients), 3-hop+ (total raw ore), aggregation (recipes consuming X), prerequisite chains, and out-of-scope questions the system must refuse. Hop labels are correct by construction; phrasing hand-checked once. Baselines: vector-only over identical chunks and graph-only, same generator, same grader. Grading: exact match on quantities and sets; Claude-judged with a strict rubric on the few prose answers, spot-checked by hand. `ragtorio bench` writes `docs/benchmark.md`: accuracy by hop count per system, p50/p95 latency, cost per query.

**Exit.** N ≥ 60. The curve exists: near parity at 1 hop, hybrid ahead at 3+. If it is not ahead, that is the result and it gets written up as such.

### Phase 8 - README and polish (1 day)

**Work.** README with the benchmark table first, the architecture diagram, license and attribution, version stance, known hard classes (ratio questions), and a short "adding a wiki" section describing the profile. `docker compose up` from a clean clone to a healthy `/health`. Security pass: no secrets in the repo, Cypher parameterised everywhere (`grep` for f-strings near `session.run`).

**Exit.** A clean clone reaches `/health` in one command. README opens with numbers.

---

## 7. Milestones and estimate

| Milestone | After phase | ~Day | Demonstrable |
|---|---|---|---|
| M1 | 3 | 9.5 | Multi-hop recipe question answered from the graph, verified against game data |
| M2 | 6 | 16 | Hybrid answers with validated citations over HTTP |
| M3 | 7 | 19 | Benchmark curve, hybrid vs vector-only by hop count |
| Done | 8 | 20 | Reproducible from clean clone |

The 20 days assume full working days; at half time it is 8 weeks. If it must shrink further, drop `tier_compare` (0.5 day) and the graph-only baseline (0.5 day). Do not drop the benchmark; it is the headline.

## 8. Stretch goals, in the order they earn their keep

1. **Stardew Valley profile.** Inline infobox, `{{Name|Iron Ore|5}}` ingredient grammar, no language subpages, no archive namespace. One profile plus one parser (`name_template_list`). Not benchmarked unless the ground-truth adapter (`Content/Data/CraftingRecipes.json`) is also written.
2. **LLM gap-fill extraction.** Relations that live only in prose: planet availability (`AVAILABLE_IN`), spoilage and recycling (`TRANSFORMS_INTO`). Strict schema via structured outputs, Message Batches API, results validated against existing entity ids, `confidence < 1.0`. The schema already has the field.
3. **Images.** Sprites linked to entities by file title (deterministic, no model), `DEPICTED_BY` edges, sprites attached to answers. Captioning of screenshots is a further step behind that.
4. **Lua module extractor.** `Module:*/data` tables are where Terraria, Minecraft and OSRS keep structured data. Needed for any of those; not for Factorio or Stardew.
5. **Incremental updates** via `list=recentchanges`.
6. **Version-diff corpus.** Ingest the `Version history` pages as a dated second corpus so "what changed in 2.0" is answerable.

## 9. Model and API decisions

- **Answering:** `claude-opus-5`, adaptive thinking on (default), streaming, cached system prefix, `effort: medium` to start.
- **Routing:** `claude-haiku-4-5`, `max_tokens` ~256, strict enum via structured outputs. Runs on every request; it must be cheap.
- **Benchmark judge (prose answers only):** `claude-opus-5` with a strict rubric, hand spot-checked.
- **Structured outputs** (`output_config.format`, `client.messages.parse()`) wherever a schema exists. No free-form JSON parsing.
- **Refusal handling:** `fallbacks: "default"` on Opus 5 requests; check `stop_reason` before reading content.
- **Embeddings:** local `bge-base-en-v1.5` by default. Not an Anthropic product; say so.
- **SDK:** `anthropic>=1,<2`. Never raw HTTP.
- **No model calls during ingestion in v1.** The graph is deterministic. This is a feature, and the README says so.

## 10. Repository layout

```
ragtorio/
  IMPLEMENTATION_PLAN.md   README.md   pyproject.toml   docker-compose.yml   .env.example
  wikis/         factorio.yaml   factorio.aliases.yaml
  src/ragtorio/
    cli.py                 ragtorio probe | harvest | extract | graph | index | ask | bench
    config.py              settings + profile loader (pydantic)
    harvest/    client.py  crawl.py  store.py
    extract/    models.py  base.py  template.py  parsers/{recipe_expr,lists,scalars}.py
    ontology/   resolve.py  load.py  check.py  schema.cypher
    index/      chunk.py  embed.py
    retrieve/   router.py  graph.py  vector.py  merge.py  cypher/*.cypher
    answer/     render.py  generate.py  validate.py
    api/        app.py  schemas.py
    bench/      ground_truth.py  generate.py  run.py  report.py
  tests/        fixtures/wikitext/factorio/*.txt   unit/   integration/
  data/         gitignored: ground_truth/
  docs/         coverage/  examples/  benchmark.md
```

Files stay under 400 lines.

## 11. Where this will go wrong

- **Trusting the extension list.** SMW is installed and empty. `ragtorio probe` checks for populated data, not installed extensions.
- **Missing namespace 3002.** No infoboxes, no graph.
- **Ingesting `/de`, `/ja`, `/zh` subpages** and building a 20x graph.
- **A page is not a node.** Items, recipes and processes do not map one-to-one to pages.
- **Serving `Archive:` content as current.**
- **Guessing `recipe` grammar edge cases.** Read the six fixtures first; multi-output recipes with probabilities and `Time` handling are where the parser breaks.
- **Model-emitted Cypher.** Banned.
- **Forgetting `internal_name`.** Without it the benchmark join to `data.raw` is a fuzzy title match and the accuracy numbers become arguable.
- **Forgetting the NC clause** in the README.

## 12. First week

1. Phase 0 in full: scaffold, compose, `ragtorio probe`, fixtures, `prototype-type` value set, finished profile.
2. Phase 1 harvester over namespaces 0, 3002 and 3004. Check the counts against the expected ranges.
3. `factorio_recipe_expr` and `plus_list`, test-first, against the fixtures.
4. Begin `TemplateExtractor`; run the first coverage report by the end of the week.

Everything after that is widening, not rearchitecting.
