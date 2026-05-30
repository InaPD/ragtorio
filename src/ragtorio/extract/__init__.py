"""Turning wikitext templates into facts.

A wiki's structured data lives in its infobox templates (see ``config.py``'s ontology
notes). This package walks from that raw wikitext to :class:`~ragtorio.extract.models.Fact`
rows, profile-driven so nothing here is Factorio-specific.
"""
