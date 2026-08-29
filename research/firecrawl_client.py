"""Firecrawl REST client for Caedral Voice research acquisition.

SECRET POLICY (program directive 2026-08-22):
- The API key lives ONLY in Windows USER env var FIRECRAWL_API_KEY.
- This module reads it at call time (process env first, then HKCU registry).
- The key is NEVER printed, logged, committed, or embedded in prompts/files.
- Subprocesses inherit it programmatically via env= dicts built by callers,
  not via shell echo.

Usage (CLI):
  python research/firecrawl_client.py search "query here" [--limit 5]
  python research/firecrawl_client.py scrape <url>
  python research/firecrawl_client.py map <url> [--limit 200]
  python research/firecrawl_client.py crawl-start <url> --limit 50
  python research/firecrawl_client.py crawl-status <job_id>

All output goes to stdout as JSON (key never included). Cache-first discipline
lives in ensure_cached() — always prefer it over raw scrape() in research code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "research" / "cache"
BASE = os.environ.get("FIRECRAWL_BASE", "https://api.firecrawl.dev")
VERSIONS = ("v2", "v1")  # try newest documented first


def _load_key() -> str:
    key = os.environ.get("FIRECRAWL_API_KEY", "").strip()
    if key:
        return key
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
        val, _ = winreg.QueryValueEx(k, "FIRECRAWL_API_KEY")
    return str(val).strip()


def _request(method: str, path: str, payload: dict | None = None,
             retries: int = 3) -> tuple[int, dict]:
    """Returns (http_status, parsed_json). Never raises on HTTP status;
    applies exponential backoff on 429/5xx per workflow policy."""
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    url = f"{BASE}{path}"
    delay = 2.0
    last_status, last_data = 0, {}
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Authorization", f"Bearer {_load_key()}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                data = json.loads(exc.read().decode("utf-8"))
            except Exception:
                data = {"error": "unparseable"}
            last_status, last_data = exc.code, data
            if exc.code == 429 or exc.code >= 500:
                time.sleep(delay)
                delay *= 2
                continue
            return exc.code, data
        except urllib.error.URLError as exc:
            last_status, last_data = -1, {"error": str(exc)}
            time.sleep(delay)
            delay *= 2
    return last_status, last_data


def _post(path: str, payload: dict):
    for version in VERSIONS:
        status, data = _request("POST", f"/{version}{path}", payload)
        if status == 200:
            return status, data, version
        if status != 404:  # non-route problems shouldn't silently fall back
            return status, data, version
    return status, data, VERSIONS[-1]


def search(query: str, limit: int = 5, scrape_markdown: bool = False):
    payload: dict = {"query": query, "limit": limit}
    if scrape_markdown:
        payload["scrapeOptions"] = {"formats": ["markdown"]}
    return _post("/search", payload)


def scrape(url: str):
    return _post("/scrape", {"url": url, "formats": ["markdown"]})


def map_site(url: str, limit: int = 500, search_filter: str | None = None):
    payload: dict = {"url": url, "limit": limit}
    if search_filter:
        payload["search"] = search_filter
    return _post("/map", payload)


def crawl_start(url: str, limit: int = 50):
    return _post("/crawl", {"url": url, "limit": limit})


def crawl_status(job_id: str):
    for version in VERSIONS:
        status, data = _request("GET", f"/{version}/crawl/{job_id}")
        if status != 404:
            return status, data, version
    return status, data, VERSIONS[-1]


# ---------- cache-first layer ----------

def _cache_path(url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    safe = re_slug(url)[:60]
    return CACHE_DIR / f"{safe}-{digest}.md"


def re_slug(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-._") else "-" for c in text]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "page"


def ensure_cached(url: str, force: bool = False) -> dict:
    """Scrape-and-cache with dedupe. Returns metadata dict (no key material)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(url)
    if path.exists() and not force:
        return {"url": url, "cache": str(path), "status": "cached",
                "bytes": path.stat().st_size}
    status, data, version = scrape(url)
    if status == 200 and data.get("data"):
        markdown = data["data"].get("markdown", "")
        meta = data["data"].get("metadata", {})
        header = (
            "---\n"
            f"url: {url}\n"
            f"title: {meta.get('title', '')}\n"
            f"retrieved_at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
            f"api_version: {version}\n"
            "---\n\n"
        )
        path.write_text(header + markdown, encoding="utf-8")
        return {"url": url, "cache": str(path), "status": "fetched",
                "bytes": path.stat().st_size}
    return {"url": url, "status": f"http_{status}", "error": str(data)[:300]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("--limit", type=int, default=5)
    sc = sub.add_parser("scrape"); sc.add_argument("url"); sc.add_argument("--force", action="store_true")
    mp = sub.add_parser("map"); mp.add_argument("url"); mp.add_argument("--limit", type=int, default=500); mp.add_argument("--filter", default=None)
    cs = sub.add_parser("crawl-start"); cs.add_argument("url"); cs.add_argument("--limit", type=int, default=50)
    cq = sub.add_parser("crawl-status"); cq.add_argument("job_id")

    args = ap.parse_args()
    out: dict
    if args.cmd == "search":
        status, data, ver = search(args.query, args.limit)
        links = [{"title": r.get("title"), "url": r.get("url"),
                  "description": (r.get("description") or "")[:160]}
                 for r in data.get("data", {}).get("web", data.get("data", []))
                 ] if isinstance(data.get("data"), dict) else \
                [{"title": r.get("title"), "url": r.get("url")} for r in (data.get("data") or [])]
        out = {"status": status, "api_version": ver, "results": links}
    elif args.cmd == "scrape":
        out = ensure_cached(args.url, force=args.force)
    elif args.cmd == "map":
        status, data, ver = map_site(args.url, args.limit, args.filter)
        # v2 shape: {"success":..., "id":..., "links":[...]} at top level;
        # v1 shape: {"data":{"links":[...]}}
        links = []
        if isinstance(data, dict):
            if isinstance(data.get("links"), list):
                links = data["links"]
            elif isinstance(data.get("data"), dict):
                links = data["data"].get("links", [])
        out = {"status": status, "api_version": ver,
               "count": len(links),
               "links": [l.get("url") if isinstance(l, dict) else l for l in links][:args.limit]}
    elif args.cmd == "crawl-start":
        status, data, ver = crawl_start(args.url, args.limit)
        out = {"status": status, "api_version": ver, "job": data}
    else:
        status, data, ver = crawl_status(args.job_id)
        out = {"status": status, "api_version": ver, "job": {
            "state": data.get("state") if isinstance(data, dict) else None,
            "completed": data.get("completed", None) if isinstance(data, dict) else None}}
    print(json.dumps(out, indent=2))
