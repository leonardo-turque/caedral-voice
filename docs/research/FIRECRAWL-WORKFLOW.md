# FIRECRAWL WORKFLOW — Caedral Voice research acquisition

Policy effective 2026-08-22 (program directive): ALL web research goes through
Firecrawl. OpenRouter web search is retired except by explicit human
reauthorization. The one-time bootstrap search was executed 2026-08-22.

## Secret handling (absolute)

- Key lives ONLY in Windows USER env var `FIRECRAWL_API_KEY` (~1 day TTL).
- `research/firecrawl_client.py` reads it at call time (process env → HKCU
  registry). It is never printed, logged, committed, or embedded in prompts.
- Verified 2026-08-22: repo-wide scan for key material = **0 files** [MEASURED].

## Flow

```
ensure_cached(url)  → cached locally? reuse : POST /scrape → research/cache/*.md
search(query)       → URL discovery only (no scrapeOptions unless needed)
map_site(root)      → site enumeration, then SELECT before scraping
crawl_start/status  → whole-site jobs ONLY when truly necessary (1 credit/page)
```

Credit discipline: Search→Map→select→Scrape is the default path; Crawl is the
exception requiring coordinator justification. Every fetched page lands in
`research/cache/<slug>-<hash>.md` with front-matter (url, title, retrieved_at,
api_version) and an entry in WEB-SOURCE-INDEX.md.

## Dedupe / sharing rules

1. Before ANY call, grep `research/cache/` and WEB-SOURCE-INDEX.md for the URL.
2. Subagents do NOT acquire the same source twice: one agent fetches, all reuse.
3. The coordinator owns WEB-SOURCE-INDEX.md rows; agents propose additions in
   their reports.

## Failure policy

| Signal | Action |
|---|---|
| 401/403 | key expired → log pending searches in EXECUTION-QUEUE.md, continue offline work |
| 429 | client auto-backoffs (exp, 3 tries); if exhausted, wait and retry later |
| timeout/5xx | limited retries (built-in), else defer; NEVER interpreted as license to use OpenRouter search |
| any outage | local-only work continues; pending list maintained |

## Validation performed (MEASURED, 2026-08-22)

| Step | Result |
|---|---|
| Real `/v2/map` on docs.firecrawl.dev | HTTP 200, 400 links |
| Real `/v2/scrape` ×8 doc pages | HTTP 200, ~150 KB cached under research/cache/ |
| Real `/v2/search` ("UTMOS pip") | HTTP 200; resolved program backlog: `pip install utmos` provides UTMOS predictor (CPU-capable, unofficial pkg) |
| Secret scan of repo | 0 files contain key material |

API notes learned from its own docs (cached): v2 responses put `links` at top
level for map; `.md` suffix on Mintlify docs yields clean markdown; errors
catalog: 429 rate/concurrency, 402 payment, 408 timeout.
