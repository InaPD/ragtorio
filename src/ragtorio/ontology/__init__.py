"""Entity resolution and the graph.

Turns the wiki-shaped facts from ``extract/`` into a wiki-agnostic graph: canonical
node ids, aliases from redirects and community shorthand, the Item/Recipe split for a
page that carries both, and edges resolved to real nodes or logged as unresolved
rather than invented.
"""
