# Phase 5 - the four templates, against the real graph

Run on 2026-09-13 against the loaded Factorio graph (867 nodes, 2,921 edges) from the
same crawl as the Phase 2, 3 and 4 numbers. The plan's exit criterion is three
hand-picked questions per template; all twelve are below with what they returned.

The equivalent checks against a purpose-built fixture graph are in
`tests/integration/test_retrieve_graph.py`. Those prove the Cypher and run in CI;
these prove the data, and needed a crawl.

## recipe_tree

| Question | Result |
|---|---|
| raw ore for one electronic circuit | Copper ore 1.5, Iron ore 1 |
| raw materials for one advanced circuit | Coal 1, Copper ore 5, Iron ore 2, Petroleum gas 20 |
| raw materials for one express transport belt | Heavy oil 20, Iron ore 31.5 |

All three check out by hand. The advanced circuit is the useful one to follow: 2 plastic
bar (one batch: 20 petroleum gas + 1 coal), 4 copper cable (2 copper plate), and 2
electronic circuits (3 copper cable -> 1.5 copper plate, plus 2 iron plate).

**Two bugs had to be fixed to get here**, both found by running the template rather than
by reading it.

*Amount-less ingredients crashed it.* 65 of 847 `CONSUMES` edges carry no amount - they
come from an item's `consumers` field, which says something consumes it without saying
how much - and multiplying `None` by a batch count raised a `TypeError`. Those branches
are now skipped rather than assumed to be one unit.

*Cycles made the arithmetic explode.* The walk had a depth bound but no cycle
detection, and Factorio's fluids are full of cycles - sulfuric acid needs water, water
is produced by recipes that need sulfuric acid. One processing unit came out costing
**twenty million sulfuric acid** before the recursion gave up at depth 16. Tracking the
items already on the current branch fixed it; the same query now returns depth 10 and
52 nodes.

*And one data-modelling rule had to change.* The ontology says "an Item with no inbound
`PRODUCES` edge is raw". Space Age added synthesis recipes for things you mine, so the
rule now misfires on seven items, and a tree that trusted it priced one processing unit
at 377 petroleum gas instead of 47. The profile carries a derived `raw_items` list -
every Item in `Category:Resources` with an inbound `PRODUCES` edge, plus water and
steam, which carry no category. The query that produces the list is in the profile
comment, so it can be re-run after a crawl rather than maintained by memory.

## unlock_chain

| Question | Result |
|---|---|
| what research do I need before a rocket silo | Rocket silo (research), 11-deep chain from Electronics |
| what gates a processing unit | Processing unit (research), via Plastics and Advanced circuit |
| prerequisites for kovarex enrichment process | Kovarex enrichment process (research), via Production science pack |

This template did not work at all until the wiki was re-read. **The graph had exactly
one `REQUIRES` edge in it.**

The profile mapped `required-technologies`, which is present on 226 of 228 technology
pages and reads exactly like a prerequisite list. Measured across the whole crawl, not
one of its 696 entries names a technology - they are all science packs and items, and
Phase 3's "both ends must be an `Unlock`" filter correctly dropped every one of them.
The technology tree is in `allows`, on 179 pages, which Phase 0 had written off as a
duplicate of `required-technologies` on the strength of a theory rather than a count.

Mapping it took three things: a parser for the trailing level range (`Automation 2, 2`
names one technology, not a level), an inversion in the resolver (`A allows B` is the
same statement as `B requires A`), and a profile-declared suffix, because `allows`
names a technology `Flammables` while its page is `Flammables (research)`. The graph now
has **392 `REQUIRES` edges** and chains up to ten deep.

**A caveat the output does not make obvious:** this returns the longest single path to a
root, not the full prerequisite closure. "Research order" is one valid order, not the
complete set of things to research. The plan asks for a path, and that is what this is.

## consumers_of

| Question | Result |
|---|---|
| what uses sulfuric acid | 6 recipes, with amounts and stations |
| what consumes petroleum gas | Plastic bar (20), Sulfur (30), Solid fuel (unstated) |
| what is an iron gear wheel used for | 25 recipes |

The amount-less edges that `recipe_tree` has to skip are exactly the ones this template
can still use: "what uses sulfuric acid" is answerable from the relationship alone. It
renders them as "an unstated amount" rather than inventing a number.

## tier_compare

| Question | Result |
|---|---|
| rank the ores by mining time | Tungsten 5, Uranium 2, then six at 1 |
| which intermediate has the largest stack size | Space science pack 2000, then a band at 200 |
| compare the belts by stack size | everything in Logistics at 100 |

The third one is a weak demonstration and is included rather than swapped out for a
flattering one. **Only three numeric properties exist in the graph**: `stack_size`,
`crafting_time`, and `mining_time` (added in this phase - a plain decimal on all 146
pages that carry it). The comparison a player actually wants for belts is throughput,
and the wiki states it in a field the profile does not map. `health` is the obvious
fourth, on 150 pages, but arrives as `{{Quality|150|195|240|285|375}}` and needs its own
parser.

"Same kind" is the wiki's own category, which Phase 3 now puts on nodes. Categories are
broad - `Logistics` covers belts and concrete - so a seed with a broad category returns a
broad comparison. Narrower grouping would mean inventing a taxonomy the wiki does not
have.

## The router

**Not measured.** The router is one `claude-haiku-4-5` call per question and there are
no credentials on this machine: no `ANTHROPIC_API_KEY`, no `ant` CLI profile. The
implementation is complete and unit-tested against a fake client - the schema enum, the
confidence downgrade, the API-failure fallback, and the fact that entities are resolved
against the graph before any parameter is bound - but its accuracy against the
[45 labeled questions](../../wikis/factorio.routing.yaml) is unknown.

```bash
export ANTHROPIC_API_KEY=...      # or: ant auth login
ragtorio route eval factorio --show-failures
```

Exit criterion: 90% on intent and 90% on template, scored separately because they fail
differently. A question routed `vector` when it should have been `graph` returns prose
where a number was wanted, which a reader spots; the right intent with the wrong
template returns confident, well-formed facts about a different question, which a reader
does not.

## Entity resolution

Resolved against the graph in three passes - exact title, the `aliases` property Phase 3
filled from redirects and community shorthand, then the full-text index. Spot-checked:

```
'green circuit'        -> Electronic circuit             (alias)
'blue science'         -> Chemical science pack          (alias)
'rocket silo'          -> Rocket silo                    (title)
'Advanced oil processing' -> Advanced oil processing (research)  (search)
'Flurbo engine'        -> unresolved
'Widget of Doom'       -> unresolved
```

The last two matter most. The first version OR-joined the search terms, and
"Flurbo engine" - a thing that does not exist - came back as "Engine unit" with a
passing score. An invented entity that resolves anyway is worse than one that does not,
because it produces a confident answer about the wrong thing. Terms are now all
required and quoted as literals.

## A hazard, still

The integration tests now default to a separate database (`ragtorio_test`), so
`make test` no longer destroys a crawl. **Neo4j has no equivalent**: the community
edition is single-database, and the Phase 3 tests load and clear a `factorio` graph by
id prefix. Running the suite still means re-running `ragtorio graph load` afterwards.
