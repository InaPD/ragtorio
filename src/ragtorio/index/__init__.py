"""The vector half of the retrieval story.

The graph answers questions whose shape is a join - what does this consume, what
gates that research. It cannot answer "why does my oil setup back up", because that
answer is prose in an article section and was never a template parameter. This package
turns article prose into embedded, citable chunks so both halves can be retrieved for
the same question and merged in Phase 5.

Every chunk keeps the page and revision it came from, so a citation can be rendered as
a permanent ``oldid`` link, and the entity ids it mentions, so a passage can be found
by the graph's own vocabulary rather than by wording alone.
"""
