// unlock_chain: the research that gates a thing, and everything that research needs.
//
// Three ways in, because a question asks about whichever of them the player has a name
// for: the entity itself when it is a technology ("what comes before oil processing
// research"), the technology that unlocks it, or - for an item - the technology that
// unlocks the *recipe* that produces it. That last one is why an item is not enough on
// its own: UNLOCKS points at "Rocket silo (recipe)", never at "Rocket silo".
//
// The depth bound is a literal because Cypher will not take a parameter inside a
// variable-length pattern. Technology trees are shallow; 10 is past the deepest real
// chain and stops a malformed graph from walking forever.
MATCH (target {id: $entity_id})
OPTIONAL MATCH (direct:Unlock)-[:UNLOCKS]->(target)
OPTIONAL MATCH (viaRecipe:Unlock)-[:UNLOCKS]->(:Recipe)-[:PRODUCES]->(target)
WITH target,
     coalesce(direct, viaRecipe, CASE WHEN target:Unlock THEN target END) AS unlock
WHERE unlock IS NOT NULL
MATCH path = (unlock)-[:REQUIRES*0..10]->(root:Unlock)
WHERE NOT (root)-[:REQUIRES]->(:Unlock)
WITH target, unlock, path
ORDER BY length(path) DESC
LIMIT 1
RETURN target.title AS target,
       unlock.title AS unlock,
       reverse([n IN nodes(path) | n.title]) AS prerequisites,
       length(path) AS depth
