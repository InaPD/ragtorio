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
