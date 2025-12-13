"""
extract_original_news.py

Reads annotated Excel (expects a column named 'link'), fetches each URL, extracts
article title, full article text, publication date (if found), and writes results
to an output Excel.

Outputs: corporate_news_originals.xlsx

Notes:
- Primary extraction uses newspaper3k. If it fails for a URL, the script
  falls back to a BeautifulSoup heuristic extractor.
- Be polite: respect robots, rate-limit requests, and avoid heavy parallelism on small servers.
"""

import concurrent.futures
import time
import logging
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
from tqdm import tqdm

# Try import newspaper; if not available the script will still attempt BS fallback
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except Exception:
    NEWSPAPER_AVAILABLE = False

# ---------- CONFIG ----------
INPUT_XLSX = "corporate_news_notebook_output_test.xlsx"   # adjust path if needed
OUTPUT_XLSX = "corporate_news_originals.xlsx"
URL_COLUMN = "link"                                  # column in your annotated Excel with URLs
MAX_WORKERS = 6                                      # thread pool size; tune by network capacity and politeness
REQUEST_TIMEOUT = 20                                 # seconds
SLEEP_BETWEEN_REQUESTS = 0.2                         # per-thread sleep to avoid bursts
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CompanyBot/1.0; +mailto:your-email@example.com)"
}
MAX_RETRIES = 2
RETRY_BACKOFF_FACTOR = 0.5
# --------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extractor")

# Requests session with retries
def make_session():
    s = requests.Session()
    retries = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.headers.update(HEADERS)
    return s

SESSION = make_session()

# Basic domain helper
def domain_from_url(url):
    try:
        return urlparse(url).netloc
    except Exception:
        return ""

# Fallback extraction using BeautifulSoup heuristics
def extract_with_bs(html, url):
    """
    Heuristic extraction with BeautifulSoup:
    - Try <article> tag
    - Try common content selectors used by news websites
    - Fallback to selecting the largest <div> by text length
    Returns: title, text, pub_date (may be None)
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.extract()

    # Title: try meta og:title, then <title>, then h1
    title = None
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"].strip()
    if not title:
        meta_title = soup.find("meta", attrs={"name": "title"}) or soup.find("meta", attrs={"name":"og:title"})
        if meta_title and meta_title.get("content"):
            title = meta_title.get("content").strip()
    if not title and soup.title:
        title = soup.title.string.strip() if soup.title.string else None
    if not title:
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            title = h1.get_text(strip=True)

    # Publication date: look for common meta tags
    pub_date = None
    for attr in [{"name":"pubdate"}, {"name":"publishdate"}, {"name":"publication_date"}, {"property":"article:published_time"},
                 {"name":"date"}, {"itemprop":"datePublished"}]:
        tag = soup.find("meta", attrs=attr)
        if tag and tag.get("content"):
            pub_date = tag["content"].strip()
            break
    # fallback: time tag
    if not pub_date:
        time_tag = soup.find("time")
        if time_tag and (time_tag.get("datetime") or time_tag.get_text(strip=True)):
            pub_date = time_tag.get("datetime") or time_tag.get_text(strip=True)

    # Try article tag(s)
    candidates = []
    article_tag = soup.find("article")
    if article_tag:
        candidates.append(article_tag)

    # Common article container selectors used by many news sites
    selectors = [
        "div.article", "div.article-body", "div.article-content", "div.story-body",
        "div.entry-content", "div.post-content", "main.article", "main#main",
        "section.article", "div#article-body", "div#article"
    ]
    for sel in selectors:
        for node in soup.select(sel):
            candidates.append(node)

    # Also consider large <div> blocks as candidate content
    # Find all divs and measure text length
    divs = soup.find_all("div")
    divs_sorted = sorted(divs, key=lambda d: len(d.get_text(separator=" ", strip=True) or ""), reverse=True)
    # take top 3 largest divs as candidates
    for d in divs_sorted[:3]:
        candidates.append(d)

    best_text = ""
    for node in candidates:
        text = node.get_text(separator="\n", strip=True)
        # discard if very short
        if len(text) < 200:
            continue
        # select longest candidate
        if len(text) > len(best_text):
            best_text = text

    # If still empty, take body text as fallback and condense
    if not best_text:
        body = soup.body
        if body:
            best_text = body.get_text(separator="\n", strip=True)
            # attempt to keep first N characters
            if len(best_text) > 10000:
                best_text = best_text[:10000]

    # Clean up repeated whitespace
    if best_text:
        best_text = "\n".join([line.strip() for line in best_text.splitlines() if line.strip()])

    return title, best_text, pub_date

# Primary extraction using newspaper3k (recommended)
def extract_with_newspaper(url, session=None):
    """
    Use newspaper3k to download & parse article. Returns title, text, publish_date.
    Note: newspaper uses its own downloader; to keep session control we still accept failing back.
    """
    try:
        a = Article(url)
        a.download()
        a.parse()
        title = a.title or None
        text = a.text or None
        pub_date = a.publish_date.isoformat() if a.publish_date else None
        # if cleaned text is too short, return None to let fallback try
        if text and len(text) < 100:
            # treat as weak extraction and fall back
            return title, text, pub_date, "weak"
        return title, text, pub_date, "ok"
    except Exception as e:
        return None, None, None, f"error:{e}"

# Worker that processes a single URL
def process_url(row):
    """
    row: dict with at least 'url' and 'source_pdf' fields
    Returns dict with extracted fields and diagnostics
    """
    url = row.get("url") or row.get("link")
    if not url or not isinstance(url, str) or url.strip() == "":
        return {"url": url, "status_code": None, "title": None, "text": None, "publication_date": None,
                "source_domain": None, "notes": "no-url"}

    url = url.strip()
    domain = domain_from_url(url)

    result = {
        "url": url,
        "source_domain": domain,
        "status_code": None,
        "title": None,
        "text": None,
        "publication_date": None,
        "notes": ""
    }

    try:
        # First try to use newspaper3k if available
        if NEWSPAPER_AVAILABLE:
            title, text, pub_date, status = extract_with_newspaper(url)
            if status == "ok" and text:
                result.update({"title": title, "text": text, "publication_date": pub_date, "notes": "newspaper3k"})
                return result
            # if newspaper3k gave weak or error, fall through to requests+bs
            # but still record attempt
            if status.startswith("error"):
                result["notes"] += f"newspaper_err:{status};"

        # Use requests to GET the page
        resp = SESSION.get(url, timeout=REQUEST_TIMEOUT)
        result["status_code"] = resp.status_code
        if resp.status_code != 200:
            result["notes"] += f"http_status_{resp.status_code};"
            # short-circuit (still attempt BS if html present)
            if not resp.headers.get("content-type", "").startswith("text"):
                return result

        # run fallback BS extractor
        title, text, pub_date = extract_with_bs(resp.text, url)
        result.update({"title": title, "text": text, "publication_date": pub_date})
        if title or text:
            result["notes"] += "bs_fallback;"
        else:
            result["notes"] += "no_content_extracted;"
    except Exception as e:
        result["notes"] += f"exception:{repr(e)};"
    finally:
        # polite pause to avoid hammering servers (per-thread)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    return result

def run_batch(input_xlsx=INPUT_XLSX, output_xlsx=OUTPUT_XLSX, max_workers=MAX_WORKERS):
    # Read input Excel; expect column URL_COLUMN ('link') or accept 'url'
    df_in = pd.read_excel(input_xlsx, engine="openpyxl")
    # normalize column existence
    if URL_COLUMN in df_in.columns:
        url_col = URL_COLUMN
    elif "url" in df_in.columns:
        url_col = "url"
    else:
        raise KeyError(f"Input sheet must contain '{URL_COLUMN}' or 'url' column")

    rows = []
    for _, r in df_in.iterrows():
        # allow semicolon-separated links per row, split them to process individually
        raw = r.get(url_col) or ""
        if not isinstance(raw, str):
            raw = str(raw)
        parts = [p.strip() for p in raw.split(";") if p.strip()]
        if not parts:
            # append an empty placeholder
            rows.append({"url": "", "source_pdf": r.get("source_pdf", "")})
        else:
            for p in parts:
                rows.append({"url": p, "source_pdf": r.get("source_pdf", "")})

    # Deduplicate URLs but preserve mapping by index
    # We'll process unique URLs but re-expand to rows_map for final output
    unique_urls = []
    url_to_indices = {}
    for i, rr in enumerate(rows):
        u = rr["url"]
        if u not in url_to_indices:
            url_to_indices[u] = []
            unique_urls.append(u)
        url_to_indices[u].append(i)

    # Prepare placeholders for results
    results_list = [None] * len(rows)

    # Process unique URLs concurrently
    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_url, {"url": u}): u for u in unique_urls}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Fetching"):
            u = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"url": u, "status_code": None, "title": None, "text": None, "publication_date": None,
                       "source_domain": domain_from_url(u), "notes": f"worker_exception:{repr(e)}"}
            # write res into all original indices that reference this url
            for idx in url_to_indices.get(u, []):
                results_list[idx] = res.copy()

    # Build output dataframe aligned to original input rows (one row per input link cell)
    out_rows = []
    for i, rr in enumerate(rows):
        res = results_list[i] or {}
        out_rows.append({
            "url": res.get("url"),
            "status_code": res.get("status_code"),
            "title": res.get("title"),
            "text": res.get("text"),
            "publication_date": res.get("publication_date"),
            "source_domain": res.get("source_domain"),
            "notes": res.get("notes"),
            "source_pdf": rr.get("source_pdf")
        })

    df_out = pd.DataFrame(out_rows)
    # Save to Excel
    df_out.to_excel(output_xlsx, index=False)
    log.info("Saved output to %s (rows=%d)", output_xlsx, len(df_out))
    return df_out

def extract_from_results(results, output_xlsx: str = OUTPUT_XLSX, max_workers: int = MAX_WORKERS):
    """
    Extract article content for a list of result dicts OR a structured `result` dict.

    Accepts:
      - results: iterable of dicts containing at least a 'url' or 'link' key (legacy)
      - OR a dict like:
          {
            "company": [
              {"name": "...", "ticker": "...", "description": [...], "link": "..."},
              ...
            ],
            "sources": [...]
          }

    Normalizes inputs, deduplicates URLs, processes them concurrently using process_url,
    and returns a pandas.DataFrame. The output includes optional metadata columns:
    company_name, company_ticker, orig_description for easier downstream mapping.
    """
    import concurrent.futures
    from tqdm import tqdm
    import pandas as pd

    # Normalize incoming 'results' into rows with 'url' and optional metadata
    rows = []

    # If a structured dict with 'company' key, expand that
    if isinstance(results, dict):
        # Prefer 'company' list if present
        company_list = results.get("company") or results.get("companies")
        if isinstance(company_list, list):
            for c in company_list:
                # c may contain 'link' or 'url'
                url = c.get("link") or c.get("url") or c.get("href") or ""
                if not isinstance(url, str):
                    url = str(url)
                parts = [p.strip() for p in url.split(";") if p.strip()] if url else []
                if not parts:
                    rows.append({
                        "url": "",
                        "source_pdf": "",
                        "meta": {
                            "company_name": c.get("name"),
                            "company_ticker": c.get("ticker"),
                            "orig_description": c.get("description")
                        }
                    })
                else:
                    for p in parts:
                        rows.append({
                            "url": p,
                            "source_pdf": "",
                            "meta": {
                                "company_name": c.get("name"),
                                "company_ticker": c.get("ticker"),
                                "orig_description": c.get("description")
                            }
                        })
        else:
            # fallback: if dict contains direct url-like keys, treat as single-item list
            url = results.get("url") or results.get("link") or ""
            if url:
                parts = [p.strip() for p in str(url).split(";") if p.strip()]
                for p in parts:
                    rows.append({"url": p, "source_pdf": "", "meta": {}})
    else:
        # Assume iterable/list of simple items
        for r in results:
            # accept strings
            if isinstance(r, str):
                parts = [r.strip()] if r.strip() else []
                for p in parts:
                    rows.append({"url": p, "source_pdf": "", "meta": {}})
                continue

            # dict-like
            url = r.get("url") or r.get("link") or r.get("href") or ""
            if not isinstance(url, str):
                url = str(url)
            parts = [p.strip() for p in url.split(";") if p.strip()]
            meta = {
                "company_name": r.get("name") or r.get("company_name"),
                "company_ticker": r.get("ticker"),
                "orig_description": r.get("description")
            }
            if not parts:
                rows.append({"url": "", "source_pdf": r.get("source_pdf", ""), "meta": meta})
            else:
                for p in parts:
                    rows.append({"url": p, "source_pdf": r.get("source_pdf", ""), "meta": meta})

    # Deduplicate while preserving mapping to original indices
    unique_urls = []
    url_to_indices = {}
    for i, rr in enumerate(rows):
        u = rr["url"]
        if u not in url_to_indices:
            url_to_indices[u] = []
            unique_urls.append(u)
        url_to_indices[u].append(i)

    # Prepare placeholders for results
    results_list = [None] * len(rows)

    # Process unique URLs concurrently using existing process_url
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_url, {"url": u}): u for u in unique_urls}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Fetching"):
            u = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"url": u, "status_code": None, "title": None, "text": None, "publication_date": None,
                       "source_domain": domain_from_url(u), "notes": f"worker_exception:{repr(e)}"}
            for idx in url_to_indices.get(u, []):
                results_list[idx] = res.copy()

    # Build output dataframe aligned to original input rows and include metadata
    out_rows = []
    for i, rr in enumerate(rows):
        res = results_list[i] or {}
        meta = rr.get("meta", {}) or {}
        out_rows.append({
            "url": res.get("url"),
            "status_code": res.get("status_code"),
            "title": res.get("title"),
            "text": res.get("text"),
            "publication_date": res.get("publication_date"),
            "source_domain": res.get("source_domain"),
            "notes": res.get("notes"),
            "source_pdf": rr.get("source_pdf"),
            # metadata from original structured input (may be None)
            "company_name": meta.get("company_name"),
            "company_ticker": meta.get("company_ticker"),
            "orig_description": meta.get("orig_description"),
        })

    df_out = pd.DataFrame(out_rows)
    
    # ✅ FIX: Only save if a valid path is provided
    if output_xlsx:
        try:
            df_out.to_excel(output_xlsx, index=False)
            # Use the global 'log' object defined at module level, or print if not available
            if 'log' in globals():
                log.info("Saved output to %s (rows=%d)", output_xlsx, len(df_out))
            else:
                print(f"Saved output to {output_xlsx} (rows={len(df_out)})")
        except Exception as e:
            print(f"Warning: Failed to save Excel file to {output_xlsx}: {e}")
            
    return df_out

if __name__ == "__main__":
    # Run batch directly when executed
    df_out = run_batch()
    print(df_out.head())