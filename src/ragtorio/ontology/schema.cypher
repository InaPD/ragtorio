// Ragtorio graph schema. Applied with `ragtorio graph load`; safe to run repeatedly.
//
// Neo4j constrains uniqueness per label, not per node, so a node that carries two
// labels (an Item that also received :Station at load time) needs a constraint on
// each label it might hold. `id` is globally unique in practice (it embeds the wiki
// and title), so this is belt-and-braces rather than a real collision risk.

CREATE CONSTRAINT item_id IF NOT EXISTS FOR (n:Item) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT fluid_id IF NOT EXISTS FOR (n:Fluid) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT recipe_id IF NOT EXISTS FOR (n:Recipe) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT station_id IF NOT EXISTS FOR (n:Station) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT unlock_id IF NOT EXISTS FOR (n:Unlock) REQUIRE n.id IS UNIQUE;

CREATE INDEX item_title IF NOT EXISTS FOR (n:Item) ON (n.title);
CREATE INDEX fluid_title IF NOT EXISTS FOR (n:Fluid) ON (n.title);
CREATE INDEX recipe_title IF NOT EXISTS FOR (n:Recipe) ON (n.title);
CREATE INDEX station_title IF NOT EXISTS FOR (n:Station) ON (n.title);
CREATE INDEX unlock_title IF NOT EXISTS FOR (n:Unlock) ON (n.title);

CREATE INDEX item_internal_name IF NOT EXISTS FOR (n:Item) ON (n.internal_name);
CREATE INDEX fluid_internal_name IF NOT EXISTS FOR (n:Fluid) ON (n.internal_name);

// Alias lookup (Phase 5's entity resolution before a query) needs "does this list
// contain X", not an exact-match index; a full-text index is what backs that.
CREATE FULLTEXT INDEX entity_aliases IF NOT EXISTS FOR (n:Item|Fluid|Recipe|Station|Unlock) ON EACH [n.title, n.aliases];
