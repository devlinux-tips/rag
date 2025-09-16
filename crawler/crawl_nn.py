#!/usr/bin/env python3
"""
Narodne Novine Crawler (polite, configurable)

Features
- Iterates a given year and issue numbers (broj) descending
- Parses listing pages to collect article HTML links
- Downloads raw HTML to an organized output folder (year/broj)
- Rate-limited requests + robots.txt compliance + retry with backoff

Usage
  python crawler/crawl_nn.py --year 2025 --start-number 120 --min-number 1 \
    --output crawler/data --delay 2.0

Notes
- This crawler is conservative: one request at a time, default 2s delay.
- Saves files as: <output>/<year>/<broj>/<slug>.html and .pdf when available
  (e.g., 2025_09_120_1707.html / .pdf).
- Skips PDF downloads for years < 2023 to avoid known 404 responses.
- Writes per-issue manifest at <output>/<year>/<broj>/manifest.jsonl.
- Only HTML pages under /clanci/sluzbeni/ are fetched (kategorija=1).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable, Optional
from urllib.parse import urlencode, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests


BASE = "https://narodne-novine.nn.hr"
LISTING_PATH = "/search.aspx"


def build_listing_url(year: int, broj: int, page: int = 1, rpp: int = 200) -> str:
    params = {
        "sortiraj": 4,
        "kategorija": 1,  # Službeni list
        "godina": year,
        "broj": broj,
        "rpp": rpp,
        "qtype": 1,
        "pretraga": "da",
    }
    if page > 1:
        params["str"] = page
    return f"{BASE}{LISTING_PATH}?{urlencode(params)}"


ARTICLE_HREF_RE = re.compile(r'href="(/clanci/sluzbeni/[^"#]+?\.html)"', re.IGNORECASE)
PAGINATION_RE = re.compile(r"[?&]str=(\d+)")
PDF_HREF_RE = re.compile(r'href="([^"]+?\.pdf)"', re.IGNORECASE)


def derive_pdf_url_from_html_url(html_url: str) -> Optional[str]:
    """Derive the canonical ELI PDF URL from an article HTML URL.

    Example:
      HTML: https://narodne-novine.nn.hr/clanci/sluzbeni/2025_09_120_1707.html
      PDF:  https://narodne-novine.nn.hr/eli/sluzbeni/2025/120/1707/pdf
    """
    try:
        name = Path(urlparse(html_url).path).stem  # e.g., 2025_09_120_1707
        m = re.match(r"^(\d{4})_(\d{2})_(\d{1,3})_(\d+)$", name)
        if not m:
            return None
        year_s, _month_s, broj_s, doc_id = m.groups()
        return f"{BASE}/eli/sluzbeni/{int(year_s)}/{int(broj_s)}/{doc_id}/pdf"
    except Exception:
        return None


@dataclass
class HttpConfig:
    delay: float = 2.0
    timeout: float = 20.0
    max_retries: int = 3
    backoff: float = 1.8
    user_agent: str = (
        "Mozilla/5.0 (compatible; NN-Crawler/0.1; +https://example.invalid)"
    )


class PoliteFetcher:
    def __init__(self, http: HttpConfig) -> None:
        self.http = http
        self._last_request_at = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": http.user_agent})

        robots_url = urljoin(BASE, "/robots.txt")
        self.robots = RobotFileParser()
        try:
            self.robots.set_url(robots_url)
            self.robots.read()
        except Exception:
            # If robots cannot be fetched, default to allow but keep delay
            self.robots = None  # type: ignore

    def allowed(self, url: str) -> bool:
        if not self.robots:
            return True
        # RobotFileParser ignores query; that's acceptable for our use.
        return self.robots.can_fetch(self.session.headers.get("User-Agent", "*"), url)

    def _respect_rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self._last_request_at
        wait = self.http.delay - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.time()

    def get(self, url: str) -> Optional[requests.Response]:
        if not self.allowed(url):
            print(f"robots.txt disallows: {url}")
            return None

        attempt = 0
        backoff = self.http.backoff
        while attempt < self.http.max_retries:
            self._respect_rate_limit()
            try:
                resp = self.session.get(url, timeout=self.http.timeout)
            except requests.RequestException as e:
                attempt += 1
                time.sleep(backoff)
                backoff *= 1.6
                if attempt >= self.http.max_retries:
                    print(f"Request failed for {url}: {e}")
                    return None
                continue

            if resp.status_code in (429, 500, 502, 503, 504):
                attempt += 1
                time.sleep(backoff)
                backoff *= 1.6
                continue

            if resp.ok:
                return resp
            else:
                print(f"HTTP {resp.status_code} for {url}")
                return None
        return None


def extract_article_links(html: str) -> list[str]:
    # Extract article hrefs
    links = [urljoin(BASE, h) for h in ARTICLE_HREF_RE.findall(html)]
    # De-duplicate while preserving order
    seen = set()
    unique = []
    for l in links:
        if l not in seen:
            unique.append(l)
            seen.add(l)
    return unique


def detect_max_page(html: str) -> int:
    pages = [int(n) for n in PAGINATION_RE.findall(html)]
    return max(pages) if pages else 1


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slug_from_url(url: str) -> str:
    """Return a filesystem-friendly name for the article (e.g., 2025_09_120_1707.html)."""
    parsed = urlparse(url)
    name = Path(parsed.path).name
    # Fallback if name is empty
    return name or re.sub(r"\W+", "_", parsed.path.strip("/")) + ".html"


def crawl_issue(fetcher: PoliteFetcher, year: int, broj: int, out_dir: Path) -> int:
    listing_url = build_listing_url(year, broj)
    print(f"Listing: {listing_url}")
    resp = fetcher.get(listing_url)
    if not resp:
        return 0

    html = resp.text
    max_page = detect_max_page(html)
    all_links = extract_article_links(html)

    # Iterate pagination if present
    for page in range(2, max_page + 1):
        page_url = build_listing_url(year, broj, page=page)
        print(f"Listing page {page}: {page_url}")
        r = fetcher.get(page_url)
        if not r:
            continue
        all_links.extend(extract_article_links(r.text))

    # Save each article
    saved = 0
    issue_dir = out_dir / str(year) / f"{broj:03d}"
    ensure_dir(issue_dir)
    manifest_path = issue_dir / "manifest.jsonl"
    manifest_entries: list[dict] = []

    # De-duplicate again after pagination
    seen = set()
    for link in all_links:
        if link in seen:
            continue
        seen.add(link)

        art = fetcher.get(link)
        if not art:
            continue

        fname = slug_from_url(link)
        fpath = issue_dir / fname
        try:
            fpath.write_text(art.text, encoding="utf-8")
            saved += 1
            print(f"Saved: {fpath}")
        except Exception as e:
            print(f"Failed to save {link} -> {fpath}: {e}")
            continue

        pdf_url: Optional[str] = None
        if year >= 2023:
            # Determine the PDF URL (prefer canonical ELI path inferred from HTML slug)
            pdf_url = derive_pdf_url_from_html_url(link)
            if not pdf_url:
                try:
                    for href in PDF_HREF_RE.findall(art.text):
                        candidate = urljoin(BASE, href)
                        if urlparse(candidate).netloc.endswith("narodne-novine.nn.hr"):
                            pdf_url = candidate
                            break
                except Exception:
                    pdf_url = None

        pdf_path: Optional[Path] = None
        if pdf_url:
            pdf_resp = fetcher.get(pdf_url)
            if pdf_resp and pdf_resp.ok:
                try:
                    pdf_path = issue_dir / (Path(fname).stem + ".pdf")
                    pdf_path.write_bytes(pdf_resp.content)
                    print(f"Saved: {pdf_path}")
                except Exception as e:
                    print(f"Failed to save PDF {pdf_url}: {e}")
                    pdf_url = None
            else:
                # Treat non-OK or missing response as no-PDF available
                pdf_url = None

        # Append to manifest (one entry per article)
        manifest_entries.append(
            {
                "year": year,
                "broj": broj,
                "slug": Path(fname).stem,
                "url_html": link,
                "path_html": str(fpath),
                "url_pdf": pdf_url,
                "path_pdf": str(pdf_path) if pdf_path else None,
            }
        )
    # Write manifest.jsonl once per issue
    try:
        with manifest_path.open("w", encoding="utf-8") as mf:
            for entry in manifest_entries:
                mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Manifest written: {manifest_path}")
    except Exception as e:
        print(f"Failed writing manifest {manifest_path}: {e}")

    return saved


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Polite crawler for Narodne Novine listings")
    p.add_argument("--year", type=int, required=True, help="Target year (godina)")
    p.add_argument(
        "--start-number",
        type=int,
        required=True,
        help="Starting issue number (broj) to count down from",
    )
    p.add_argument(
        "--min-number",
        type=int,
        default=1,
        help="Minimum issue number (inclusive) to stop at",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("crawler/data"),
        help="Output directory for saved HTML",
    )
    p.add_argument("--delay", type=float, default=2.0, help="Delay between requests in seconds")
    p.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds")
    p.add_argument("--max-retries", type=int, default=3, help="Max HTTP retries on failure")
    p.add_argument("--user-agent", type=str, default=None, help="Custom User-Agent string")

    args = p.parse_args(list(argv) if argv is not None else None)

    if args.start_number < args.min_number:
        print("start-number must be >= min-number", file=sys.stderr)
        return 2

    http_cfg = HttpConfig(
        delay=args.delay,
        timeout=args.timeout,
        max_retries=args.max_retries,
        user_agent=(args.user_agent or HttpConfig.user_agent),
    )
    fetcher = PoliteFetcher(http_cfg)

    ensure_dir(args.output)

    total_saved = 0
    for broj in range(args.start_number, args.min_number - 1, -1):
        print("-" * 72)
        print(f"Year {args.year} — Issue broj {broj}")
        try:
            saved = crawl_issue(
                fetcher,
                args.year,
                broj,
                args.output,
            )
            total_saved += saved
            print(f"Issue {broj}: saved {saved} pages")
        except KeyboardInterrupt:
            print("Interrupted by user.")
            break
        except Exception as e:
            print(f"Error on broj {broj}: {e}")
            continue

    print("=" * 72)
    print(f"Done. Total pages saved: {total_saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
