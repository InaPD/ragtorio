# Ragtorio

Knowledge-graph and vector RAG over the [Factorio wiki](https://wiki.factorio.com), built
so a second crafting-game wiki is a config file rather than a rewrite.

Multi-hop questions players actually ask - *what raw ore does a rocket silo cost*, *what
research do I need before oil processing* - are answered from a knowledge graph extracted
**deterministically from wiki templates**, with mechanics prose retrieved alongside from a
vector index. No model touches the ingestion pipeline, so no edge in the graph is
hallucinated.

> **Status: Phase 0 of 9 complete.** Scaffolding, wiki profile, and the `probe` command.
> The harvester, extractor, graph, index, retrieval and benchmark are not built yet.
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
```

Copy `.env.example` to `.env` and set `RAGTORIO_CONTACT_EMAIL` before any crawl - wiki operators
expect a working contact address in the User-Agent.

## Adding a wiki

A wiki is described by one YAML file in [`wikis/`](wikis/), validated on load. Nothing in
`src/` is Factorio-specific.

```bash
ragtorio probe https://example.wiki/api.php     # what does it run, and is the data real?
cp wikis/factorio.yaml wikis/newwiki.yaml # edit namespaces, fields, type map
ragtorio profile newwiki                        # fails loudly on a typo
```

If the new wiki's infobox grammar differs, add one pure function to
`extract/parsers/` and name it in the profile.

## Layout

```
wikis/factorio.yaml          the whole Factorio-specific surface
src/ragtorio/
  cli.py                     ragtorio probe | profile  (harvest, extract, ... to come)
  config.py                  profile schema and loader, validated with pydantic
  harvest/client.py          rate-limited, retrying MediaWiki client
  harvest/probe.py           installed-versus-populated structured-data check
tests/fixtures/wikitext/     committed wikitext, so extraction tests need no network
docs/coverage/               what the wiki actually contains
scripts/                     one-off Phase 0 measurement tools
```

## Licensing

Wiki content is **CC BY-NC-SA 3.0** (Factorio Wiki, Wube Software). The non-commercial
clause is real: this is a portfolio and research project, not a basis for a paid product.
Each wiki profile carries its own license and attribution string, because they differ -
Fandom is CC BY-SA with no NC clause.

The code in this repository is MIT.
