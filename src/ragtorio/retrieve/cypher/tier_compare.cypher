// tier_compare: things of the same kind, ordered by a number.
//
// "Same kind" is the wiki's own category, which is the only grouping a player would
// recognise and the reason Phase 3 puts categories on nodes at all. The seed entity
// supplies the categories; every peer sharing one and carrying the property competes.
//
// Peers are collected once and their shared categories aggregated, rather than one row
// per shared category: two items both filed under "Intermediate products" and
// "Resources" are one comparison, not two, and returning them twice would read to the
// answering model as two different items with the same name.
//
// The property is a parameter used through dynamic property access rather than being
// spliced into the query text - the one place a template needs a column name at
// runtime, and the one place it would be tempting to build Cypher from a string.
MATCH (seed {id: $entity_id})
WITH coalesce(seed.categories, []) AS seedCategories
MATCH (peer)
WHERE peer.id STARTS WITH $prefix
  AND peer[$property] IS NOT NULL
  AND any(c IN coalesce(peer.categories, []) WHERE c IN seedCategories)
RETURN peer.title AS title,
       peer[$property] AS value,
       [c IN peer.categories WHERE c IN seedCategories] AS categories
ORDER BY value DESC, title
LIMIT $limit
