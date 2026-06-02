"""Deciding what to fetch for a question, fetching it, and merging the two halves.

The graph and the vector index answer different shapes of question, and a question
does not announce which shape it is. This package is the part that decides: a small,
cheap model reads the question and returns an intent, a query template and the
entities it names, all three constrained to a schema. Nothing here lets a model write
a query - the templates are files, the parameters are bound, and an entity is resolved
against the graph's own vocabulary before any query runs.
"""
