import os
import json
import time
import datetime
import requests
from tqdm import tqdm

USER_AGENT = (
    "cs2881r-mini-project/0.1 "
    "(https://github.com/evangeloskassos/cs2881r-mini-project; "
    "contact=github@evangeloskassos.com)"
)
API_URL = "https://en.wikipedia.org/w/api.php"

DEFAULT_DIR = "data"
DEFAULT_FILE = os.path.join(DEFAULT_DIR, "articles.jsonl")


def ensure_data_dir():
    os.makedirs(DEFAULT_DIR, exist_ok=True)


def save_to_jsonl(records, path=DEFAULT_FILE):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")


def get_recent_wikipedia_articles(limit=500, min_bytes=3000):
    """
    Fetch recently created Wikipedia articles using 'recentchanges' API.
    Continues fetching pages until at least `limit` results meet the min_bytes threshold.
    Returns a list of dicts with {title, timestamp, newlen}.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    three_months_ago = now - datetime.timedelta(days=90)
    rccontinue = None
    results = []

    with requests.Session() as s:
        s.headers["User-Agent"] = USER_AGENT
        while len(results) < limit:
            params = {
                "action": "query",
                "format": "json",
                "list": "recentchanges",
                "rctype": "new",
                "rcnamespace": 0,
                "rcdir": "newer",
                "rclimit": 500,
                "rcstart": three_months_ago.isoformat(),
                "rcend": now.isoformat(),
                "rcprop": "title|timestamp|sizes|ids",
            }
            if rccontinue:
                params["rccontinue"] = rccontinue

            r = s.get(API_URL, params=params)
            r.raise_for_status()
            data = r.json()

            changes = data.get("query", {}).get("recentchanges", [])
            rccontinue = data.get("continue", {}).get("rccontinue")

            for item in changes:
                newlen = item.get("newlen", 0)
                if newlen >= min_bytes:
                    results.append(
                        {
                            "title": item.get("title"),
                            "pageid": item.get("pageid"),
                            "timestamp": item.get("timestamp"),
                            "newlen": newlen,
                        }
                    )

            if not rccontinue:
                break
            time.sleep(0.3)

    return results[:limit]


def confirm_by_word_count(articles, min_words=100, batch_size=20):
    """
    Confirms filtered articles by fetching their text and counting words.
    Handles pagination to ensure all pages are fetched.
    """
    confirmed = []
    with requests.Session() as s:
        s.headers["User-Agent"] = USER_AGENT

        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]
            pageids = [a["pageid"] for a in batch]
            seen_pageids = set()

            for pid in pageids:
                params = {
                    "action": "query",
                    "format": "json",
                    "prop": "extracts",
                    "explaintext": True,
                    "pageids": str(pid),
                }

                r = s.get(API_URL, params=params)
                r.raise_for_status()
                data = r.json()
                pages = data.get("query", {}).get("pages", {})

                for p in pages.values():
                    if p["pageid"] in seen_pageids:
                        continue
                    seen_pageids.add(p["pageid"])
                    extract = p.get("extract", "")
                    if extract:
                        wc = len(extract.split())
                        if wc >= min_words:
                            confirmed.append(
                                {
                                    "title": p["title"],
                                    "pageid": p["pageid"],
                                    "words": wc,
                                    "extract": extract,
                                }
                            )

            # Be polite to the API
            time.sleep(0.2)

    return confirmed


def collect_articles(target_count=1165, min_bytes=3000, min_words=100):
    """
    Collects recently created Wikipedia articles:
    - Ensures get_recent_wikipedia_articles() yields at least target_count byte-qualified results.
    - Confirms each by word count.
    - Saves continuously to JSONL file.
    """
    ensure_data_dir()
    total = []

    with tqdm(total=target_count) as pbar:
        while len(total) < target_count:
            batch = get_recent_wikipedia_articles(limit=100, min_bytes=min_bytes)
            confirmed = confirm_by_word_count(batch, min_words=min_words)
            if confirmed:
                total.extend(confirmed)
                pbar.update(len(confirmed))
            time.sleep(0.5)

    final = total[:target_count]

    save_to_jsonl(final)
    return final


if __name__ == "__main__":
    results = collect_articles(target_count=1165, min_bytes=3000, min_words=100)
    print(
        f"\n✅ Collected {len(results)} Wikipedia articles created in the last 3 months (>100 words)\n"
    )
