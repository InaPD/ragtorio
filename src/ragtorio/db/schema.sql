-- Ragtorio schema. Applied with `ragtorio init-db`; safe to run repeatedly.
--
-- Phase 1 owns the raw_* tables and crawl_run. Later phases add fact, chunk and
-- routing_log. Raw wikitext is kept verbatim so extraction can be re-run offline
-- after a parser change without touching the wiki again.

CREATE TABLE IF NOT EXISTS crawl_run (
    run_id       bigserial PRIMARY KEY,
    wiki         text        NOT NULL,
    started_at   timestamptz NOT NULL DEFAULT now(),
    finished_at  timestamptz,
    listed       integer     NOT NULL DEFAULT 0,
    translations integer     NOT NULL DEFAULT 0,
    unchanged    integer     NOT NULL DEFAULT 0,
    fetched      integer     NOT NULL DEFAULT 0,
    redirects    integer     NOT NULL DEFAULT 0,
    error        text
);

CREATE INDEX IF NOT EXISTS crawl_run_wiki_idx ON crawl_run (wiki, started_at DESC);

-- One row per page, current revision only. History is not needed: the benchmark
-- compares against one game version.
CREATE TABLE IF NOT EXISTS raw_page (
    wiki        text        NOT NULL,
    page_id     bigint      NOT NULL,
    ns          integer     NOT NULL,
    title       text        NOT NULL,
    revision_id bigint      NOT NULL,
    revised_at  timestamptz NOT NULL,
    wikitext    text        NOT NULL,
    fetched_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (wiki, page_id)
);

-- Extraction walks from an article to its Infobox: page by title, so the title
-- lookup must be both unique and fast.
CREATE UNIQUE INDEX IF NOT EXISTS raw_page_title_idx ON raw_page (wiki, title);
CREATE INDEX IF NOT EXISTS raw_page_ns_idx ON raw_page (wiki, ns);

-- Redirects become node aliases in Phase 3 ("green circuit" -> "Electronic circuit").
CREATE TABLE IF NOT EXISTS raw_redirect (
    wiki       text NOT NULL,
    from_title text NOT NULL,
    to_title   text NOT NULL,
    PRIMARY KEY (wiki, from_title)
);

CREATE INDEX IF NOT EXISTS raw_redirect_target_idx ON raw_redirect (wiki, to_title);

-- Category membership, used for the archived flag and for tier comparisons.
CREATE TABLE IF NOT EXISTS raw_category (
    wiki     text   NOT NULL,
    page_id  bigint NOT NULL,
    category text   NOT NULL,
    PRIMARY KEY (wiki, page_id, category)
);

CREATE INDEX IF NOT EXISTS raw_category_name_idx ON raw_category (wiki, category);

-- Phase 2: the extractor's output. A run replaces every fact for the wiki, since
-- everything here is recomputable from raw_page without refetching. `object` and
-- `props` are jsonb so a fact's value keeps its real type (string, number, or null)
-- rather than everything collapsing to text.
CREATE TABLE IF NOT EXISTS fact (
    fact_id             bigserial PRIMARY KEY,
    wiki                text        NOT NULL,
    subject             text        NOT NULL,
    subject_labels      text[]      NOT NULL,
    predicate           text        NOT NULL,
    object              jsonb,
    object_labels       text[]      NOT NULL DEFAULT '{}',
    props               jsonb       NOT NULL DEFAULT '{}',
    source_page_id      bigint      NOT NULL,
    source_revision_id  bigint      NOT NULL,
    source_field        text        NOT NULL,
    extracted_at        timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS fact_wiki_subject_idx ON fact (wiki, subject);
CREATE INDEX IF NOT EXISTS fact_wiki_predicate_idx ON fact (wiki, predicate);

-- Phase 4: the vector index. pgvector ships in the compose image; the extension is
-- created here so a hand-rolled Postgres fails on this line rather than three
-- statements later with a confusing "type vector does not exist".
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per retrievable passage. A run replaces every chunk for the wiki, for the
-- same reason a run replaces its facts: everything here is recomputable from
-- raw_page, so after a chunker or embedding-model change the old rows are wrong
-- rather than merely stale.
--
-- `title` is denormalised from raw_page deliberately. Every citation renders as
-- index.php?title=X&oldid=N, so the title is needed on every single retrieved row,
-- and a join to fetch it would buy nothing but the chance of the two disagreeing.
--
-- The column is vector(768) because the default provider is bge-base-en-v1.5. Vector
-- width belongs to the provider, not to the schema; `ragtorio index build` widens or
-- narrows this column when a different provider is selected.
CREATE TABLE IF NOT EXISTS chunk (
    chunk_id             text        PRIMARY KEY,
    wiki                 text        NOT NULL,
    page_id              bigint      NOT NULL,
    title                text        NOT NULL,
    revision_id          bigint      NOT NULL,
    section_path         text[]      NOT NULL DEFAULT '{}',
    text                 text        NOT NULL,
    embedding            vector(768) NOT NULL,
    mentioned_entity_ids text[]      NOT NULL DEFAULT '{}',
    indexed_at           timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunk_wiki_idx ON chunk (wiki);
CREATE INDEX IF NOT EXISTS chunk_page_idx ON chunk (wiki, page_id);

-- Approximate nearest neighbour. Cosine, matching the normalised vectors every
-- provider emits; build-time defaults (m=16, ef_construction=64) are pgvector's own,
-- and query-time accuracy is tuned per query with hnsw.ef_search rather than baked in
-- here - which is what `ragtorio index recall --sweep` measures.
CREATE INDEX IF NOT EXISTS chunk_embedding_idx ON chunk USING hnsw (embedding vector_cosine_ops);

-- Phase 5 filters retrieval by the entities its router resolved, which is a
-- containment test over an array: GIN, not btree.
CREATE INDEX IF NOT EXISTS chunk_entities_idx ON chunk USING gin (mentioned_entity_ids);
