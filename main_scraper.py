#!/usr/bin/env python3

"""
main_scraper.py

Purpose
- Find recently-active bills (from Congress.gov if available, otherwise a manual source)
- For each bill: extract bill number and raw legislative text URL
- Construct CBO cost-estimate URLs and scrape CBO for a "score" and quantified language about affected/beneficiary groups
- Attempt a GAO search for related reports (best-effort)
- Emit structured JSON records per bill into an artifacts/ directory

Notes / operation
- The script tries sources in this order:
   1) Congress.gov API (requires CONGRESS_GOV_API_KEY and optional CONGRESS_GOV_API_BASE)
   2) Manual list: set BILLS_SOURCE=manual and provide RECENT_BILLS_JSON env var (JSON array of { "bill_number": "...", "raw_text_url": "...", "status": "..." })
- Environment variables:
   - BILLS_SOURCE: "congress.gov" | "manual" (default: "congress.gov")
   - CONGRESS_GOV_API_KEY, CONGRESS_GOV_API_BASE (default: https://api.congress.gov/v3)
   - TARGET_HOURS: how far back to look for activity (default 48)
   - ARTIFACTS_DIR: output directory (default "artifacts")
   - REQUEST_TIMEOUT: seconds for HTTP requests (default 15)
- The CBO URL pattern used: https://www.cbo.gov/cost-estimates/{chamber_slug}/{number}
   e.g. H.R. 1 -> https://www.cbo.gov/cost-estimates/hr/1
- This is a best-effort scraper with heuristics. Adapt the Congress.gov calls to match your available API keys and endpoints.
"""

from __future__ import annotations

import os
import re
import json
import time
import logging
import datetime
from typing import List, Dict, Optional, Any

import requests
from bs4 import BeautifulSoup

# Configuration / environment
BILLS_SOURCE = os.getenv("BILLS_SOURCE", "congress.gov").lower()
CONGRESS_GOV_API_KEY = os.getenv("CONGRESS_GOV_API_KEY")
CONGRESS_GOV_API_BASE = os.getenv("CONGRESS_GOV_API_BASE", "https://api.congress.gov/v3")
TARGET_HOURS = int(os.getenv("TARGET_HOURS", "48"))
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))
USER_AGENT = os.getenv("USER_AGENT", "unfiltered-record-audit-bot/1.0 (+https://github.com/unfilteredrecordofficial)")

HEADERS = {"User-Agent": USER_AGENT}

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def _iso_utc_now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def normalize_bill_number(bill_number: str) -> Optional[Dict[str, str]]:
    """
    Convert a bill number like "H.R. 1" or "S. 2" into a slug used by CBO:
      returns {"chamber": "hr" | "s", "number": "1"}
    Returns None on parse failure.
    """
    if not bill_number:
        return None
    cleaned = bill_number.strip().upper().replace(" ", "").replace("..", ".")
    m = re.match(r"^(H\.?.?R\.?|HR)(\-)?(\d+)$", cleaned)
    if m:
        return {"chamber": "hr", "number": m.group(3)}
    m = re.match(r"^(S\.?|S)(\-)?(\d+)$", cleaned)
    if m:
        return {"chamber": "s", "number": m.group(3)}
    m2 = re.match(r"^([A-Z]+).*?(\d+)$", bill_number.replace(".", ""))
    if m2:
        prefix = m2.group(1)
        num = m2.group(2)
        if prefix.startswith("HR") or prefix.startswith("H"):
            return {"chamber": "hr", "number": num}
        if prefix.startswith("S"):
            return {"chamber": "s", "number": num}
    return None

def construct_cbo_url(bill_number: str) -> Optional[str]:
    norm = normalize_bill_number(bill_number)
    if not norm:
        return None
    return f"https://www.cbo.gov/cost-estimates/{{norm['chamber']}}/{{norm['number']}}"

def safe_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = REQUEST_TIMEOUT) -> Optional[requests.Response]:
    headers = {**HEADERS, **(headers or {})}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r
    except requests.RequestException as ex:
        logging.debug("Request failed for %s: %s", url, ex)
        return None

def extract_cbo_score_and_groups(html: str) -> Dict[str, Any]:
    """
    Heuristic extraction from a CBO cost-estimate HTML:
      - find a primary numeric cost (e.g., $X billion, $Y million) that is labeled as cost, deficit, or savings
      - find sentences that quantify impacted groups (have a number and keywords like uninsured, households, beneficiaries)
    Returns: { "score_text": str | None, "score_value": float | None, "score_unit": "dollars"|"people"|None,
               "affected_sentences": [str, ...] }
    """
    soup = BeautifulSoup(html, "html.parser")

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s+\n", "\n\n", text)

    money_regex = re.compile(r"(\$|USD)?\s*([0-9,]+(?:\.\d+)?)\s*(million|billion|thousand)?", flags=re.IGNORECASE)
    keywords = ["cost", "estimate", "increase", "decrease", "deficit", "savings", "budget", "net", "spending"]
    best_match = None
    for m in money_regex.finditer(text):
        start = max(0, m.start() - 200)
        excerpt = text[start:m.end() + 200]
        if any(k in excerpt.lower() for k in keywords):
            best_match = (m, excerpt)
            break

    score_text = None
    score_value = None
    score_unit = None
    if best_match:
        m, excerpt = best_match
        score_text = excerpt.strip()[:1000]
        num_raw = m.group(2)
        multiplier = m.group(3) or ""
        try:
            num = float(num_raw.replace(",", ""))
            if multiplier.lower().startswith("b"):
                num *= 1_000_000_000
            elif multiplier.lower().startswith("m"):
                num *= 1_000_000
            elif multiplier.lower().startswith("th"):
                num *= 1_000
            score_value = float(num)
            score_unit = "dollars"
        except Exception:
            score_value = None

    group_keywords = [
        "uninsured",
        "household",
        "households",
        "families",
        "children",
        "seniors",
        "elderly",
        "low-income",
        "benefit",
        "beneficiaries",
        "workers",
        "employers",
        "individuals",
        "people",
    ]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    affected_sentences: List[str] = []
    for s in sentences:
        if re.search(r"\b\d[\d,\.]*\b", s) and any(k in s.lower() for k in group_keywords):
            affected_sentences.append(s.strip())
    affected_sentences = list(dict.fromkeys(affected_sentences))[:20]

    return {
        "score_text": score_text,
        "score_value": score_value,
        "score_unit": score_unit,
        "affected_sentences": affected_sentences,
    }

def find_gao_reports(bill_number: str) -> List[Dict[str, str]]:
    """
    Best-effort GAO search: use GAO search with query=bill_number and parse first few results.
    Returns a list of { "title": ..., "url": ... }.
    """
    query = bill_number.replace(" ", "+")
    search_url = f"https://www.gao.gov/search?query={{query}}"
    r = safe_get(search_url)
    if not r:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    for a in soup.select("h3 a")[:10]:
        href = a.get("href")
        if not href:
            continue
        href_full = href if href.startswith("http") else "https://www.gao.gov" + href
        title = a.get_text(strip=True)
        results.append({"title": title, "url": href_full})
    return results

def parse_congress_gov_recent(since_dt: datetime.datetime) -> List[Dict[str, Any]]:
    """
    Attempt to query Congress.gov API for bills with action since since_dt.
    Defensive handling for varying endpoint shapes.
    """
    if not CONGRESS_GOV_API_KEY:
        logging.info("No CONGRESS_GOV_API_KEY provided; skipping Congress.gov backend.")
        return []

    since_str = since_dt.strftime("%Y-%m-%d")
    params = {"lastActionDate": since_str, "api_key": CONGRESS_GOV_API_KEY}
    endpoint = f"{{CONGRESS_GOV_API_BASE}}/bills"
    logging.info("Querying Congress.gov: %s params=%s", endpoint, {"lastActionDate": since_str})
    try:
        r = requests.get(endpoint, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        candidates = []
        if isinstance(data, dict):
            if "bills" in data:
                candidates = data["bills"]
            elif "results" in data:
                candidates = data["results"]
            elif "data" in data and isinstance(data["data"], dict) and "bills" in data["data"]:
                candidates = data["data"]["bills"]
        if not candidates:
            logging.debug("Congress.gov returned no candidate bills; response keys: %s", list(data.keys()))
            return []
        out: List[Dict[str, Any]] = []
        for item in candidates:
            bill_number = item.get("number") or item.get("bill_number") or item.get("billNumber") or item.get("bill")
            title = item.get("title") or item.get("official_title") or item.get("short_title")
            status = item.get("latestAction") or item.get("status") or item.get("latest_action") or ""
            raw_text_url = None
            urls = item.get("urls") or item.get("url") or item.get("document_url") or {}
            if isinstance(urls, dict):
                raw_text_url = urls.get("text") or urls.get("raw") or urls.get("pdf")
            elif isinstance(urls, str):
                raw_text_url = urls
            if not raw_text_url:
                docs = item.get("documents") or item.get("congressdotgov_url") or []
                if isinstance(docs, list) and docs:
                    for d in docs:
                        if isinstance(d, dict):
                            for k in ("pdf_url", "document_url", "url"):
                                if d.get(k):
                                    raw_text_url = d.get(k)
                                    break
                        if raw_text_url:
                            break
            if not bill_number:
                continue
            out.append(
                {
                    "bill_number": bill_number,
                    "title": title,
                    "status": status,
                    "raw_text_url": raw_text_url,
                }
            )
        logging.info("Congress.gov backend returned %d bills", len(out))
        return out
    except Exception as e:
        logging.exception("Error querying Congress.gov: %s", e)
        return []

def load_manual_bills() -> List[Dict[str, Any]]:
    """
    Load a manual JSON list from RECENT_BILLS_JSON env var or recent_bills.json file.
    The JSON must be an array of objects: { bill_number, raw_text_url, status?, title? }
    """
    payload = os.getenv("RECENT_BILLS_JSON")
    if payload:
        try:
            data = json.loads(payload)
            logging.info("Loaded %d manual bills from RECENT_BILLS_JSON", len(data) if isinstance(data, list) else 0)
            return data if isinstance(data, list) else []
        except Exception as e:
            logging.exception("RECENT_BILLS_JSON parse failed: %s", e)
    path = os.path.join(os.getcwd(), "recent_bills.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logging.info("Loaded %d manual bills from recent_bills.json", len(data) if isinstance(data, list) else 0)
                return data if isinstance(data, list) else []
        except Exception:
            logging.exception("Failed to read recent_bills.json")
    logging.info("No manual bills provided")
    return []

def get_recent_bills(hours_back: int = TARGET_HOURS) -> List[Dict[str, Any]]:
    since_dt = datetime.datetime.utcnow() - datetime.timedelta(hours=hours_back)
    logging.info("Gathering recent bills since %s (UTC)", since_dt.isoformat())
    if BILLS_SOURCE == "manual":
        return load_manual_bills()
    # try congress.gov
    out = parse_congress_gov_recent(since_dt)
    if out:
        return out
    # last fallback: manual
    return load_manual_bills()

def compile_record(bill: Dict[str, Any]) -> Dict[str, Any]:
    bill_number = bill.get("bill_number") or bill.get("bill") or bill.get("number")
    title = bill.get("title")
    status = bill.get("status")
    raw_text_url = bill.get("raw_text_url")
    record = {
        "bill_number": bill_number,
        "title": title,
        "status": status,
        "raw_text_url": raw_text_url,
        "fetched_at": _iso_utc_now(),
    }

    # CBO
    cbo_url = construct_cbo_url(bill_number) if bill_number else None
    record["cbo_url"] = cbo_url
    record["cbo"] = None
    if cbo_url:
        logging.info("Fetching CBO for %s -> %s", bill_number, cbo_url)
        r = safe_get(cbo_url)
        if r:
            parsed = extract_cbo_score_and_groups(r.text)
            record["cbo"] = parsed
        else:
            logging.info("No CBO page found at %s", cbo_url)

    # GAO
    try:
        gao = find_gao_reports(bill_number) if bill_number else []
        record["gao_reports"] = gao
    except Exception:
        record["gao_reports"] = []

    return record

def save_record(record: Dict[str, Any], out_dir: str = ARTIFACTS_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_name = (record.get("bill_number") or "unknown").replace("/", "_").replace(" ", "_")
    filename = f"record_{{safe_name}}_{{ts}}.json"
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path

def main() -> int:
    try:
        bills = get_recent_bills(TARGET_HOURS)
        if not bills:
            logging.info("No bills found by backends. Exiting without output.")
            return 0
        logging.info("Processing %d bills", len(bills))
        saved_paths = []
        for b in bills:
            try:
                rec = compile_record(b)
                p = save_record(rec)
                saved_paths.append(p)
                # polite pause to avoid hammering sites
                time.sleep(1)
            except Exception:
                logging.exception("Failed to process bill: %s", b)
        logging.info("Saved %d records under %s", len(saved_paths), ARTIFACTS_DIR)
        return 0
    except Exception:
        logging.exception("Unexpected failure in main")
        return 2

if __name__ == "__main__":
    raise SystemExit(main())