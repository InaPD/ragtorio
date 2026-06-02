// consumers_of: everything that consumes an item, and what it turns it into.
//
// The inverse of recipe_tree's own direction, and the reason an amount-less CONSUMES
// edge is still worth having: "what uses sulfuric acid" is answerable from the
// relationship alone, even where the wiki never stated a quantity. The amount is
// returned as-is, null included, so the renderer can say "unspecified" rather than
// invent one.
MATCH (recipe:Recipe)-[c:CONSUMES]->(item {id: $entity_id})
OPTIONAL MATCH (recipe)-[:PRODUCES]->(output)
OPTIONAL MATCH (recipe)-[:CRAFTED_AT]->(station)
RETURN recipe.title AS recipe,
       c.amount AS amount,
       collect(DISTINCT output.title) AS produces,
       collect(DISTINCT station.title) AS stations
ORDER BY recipe
LIMIT $limit
