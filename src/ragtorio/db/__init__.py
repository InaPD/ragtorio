"""Postgres: the raw crawl, extracted facts, chunks and logs.

Everything downstream of the ``raw_*`` tables is recomputable without refetching, so
this is the only layer that must survive a rebuild.
"""
