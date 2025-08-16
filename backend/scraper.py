import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
import json
import os

TARGET_AUTHORS = {
    "mplus313", "rhbinvest", "AmInvest", "MalaccaSecurities",
    "sectoranalyst", "PhillipCapital", "HLInvest", "PublicInvest"
}

TARGET_DATE = "25 July 2025"  # Change as needed
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NewsScraper/1.0)"}
BASE_URL = "https://klse.i3investor.com"
IMAGE_FOLDER = "technical_charts"


def fetch_research_headlines():
    url = f"{BASE_URL}/web/headline/blog?type=research"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.text


def parse_headlines(html):
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one("#news-blog")
    results = []

    current_date = None
    for el in container.find_all(recursive=False):
        h5 = el.select_one("h5")
        if h5:
            match = re.search(r"\d{1,2} \w+ \d{4}", h5.text)
            if match:
                current_date = match.group(0)
            continue

        if el.name == "ul" and "ms-4" in el.get("class", []):
            li = el.select_one("li")
            if not li:
                continue
            a_tag = li.find("a", href=True)
            subtitle = li.select_one("span.subtitle a")
            if not a_tag or not subtitle:
                continue
            author = subtitle.text.strip()
            if author not in TARGET_AUTHORS:
                continue
            if current_date != TARGET_DATE:
                continue

            full_url = a_tag["href"] if a_tag["href"].startswith("http") else BASE_URL + a_tag["href"]

            results.append({
                "title": a_tag.text.strip(),
                "author": author,
                "date": current_date,
                "url": full_url
            })

    return results


def fetch_blog_content(url):
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    content_div = soup.select_one("#blogcontent")
    if not content_div:
        return {"text": "", "images": []}

    # Extract text content
    content_parts = []
    for tag in content_div.find_all(["h3", "p", "li"], recursive=True):
        content_parts.append(tag.get_text(strip=True))

    # Extract images
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    image_filenames = []
    for img_tag in content_div.find_all("img"):
        src = img_tag.get("src")
        if not src:
            continue
        img_url = BASE_URL + src if src.startswith("/") else src
        filename = os.path.basename(img_url)  # ✅ only save filename
        filepath = os.path.join(IMAGE_FOLDER, filename)

        try:
            img_resp = requests.get(img_url, headers=HEADERS)
            img_resp.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(img_resp.content)
            print(f"🖼️  Saved image: {filepath}")
        except Exception as e:
            print(f"⚠️  Failed to download image {img_url}: {e}")
            continue

        image_filenames.append(filename)

    return {
        "text": "\n".join(content_parts),
        "images": image_filenames
    }

def main():
    print("🔍 Fetching research headlines...")
    html = fetch_research_headlines()
    items = parse_headlines(html)
    print(f"✅ Found {len(items)} matching posts by target authors on {TARGET_DATE}.")

    for i in items:
        print(f"📝 Fetching blog content from: {i['url']}")
        blog_data = fetch_blog_content(i["url"])
        i["content"] = blog_data["text"]
        i["images"] = blog_data["images"]

    with open("scraper_output.json", "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"✅ Output saved to scraper_output.json")


if __name__ == "__main__":
    main()
