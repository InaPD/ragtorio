"""The HTTP surface: ``POST /ask`` and ``GET /health``.

Thin on purpose. Everything the API does is done by the retrieval and answer pipelines
the CLI already drives, so there is exactly one code path that answers a question and
the benchmark measures the same system a reader can curl.
"""
