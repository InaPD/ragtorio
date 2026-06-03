"""Turning retrieved context into a grounded answer, and checking that it is grounded.

Three things happen here, and they are deliberately three modules. ``render`` turns
what Phase 5 retrieved into the text a model sees, including the recipe tree as a real
nested list rather than the flat lines the retriever logs. ``generate`` makes exactly
one model call. ``validate`` reads the answer back and checks every citation against
the chunk ids that were actually retrieved.

The split exists because only the middle one costs money or needs a network. Rendering
and validation are pure functions over data the pipeline already has, which is what
lets the interesting half of this phase - what happens when a model cites a source
that was never retrieved - be tested without an API key.
"""
