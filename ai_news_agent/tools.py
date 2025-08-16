from langchain.tools import tool
from typing import Dict
import requests
from bs4 import BeautifulSoup
import re
from .config import BASE_URL, HEADERS, TARGET_AUTHORS, TARGET_DATE


@tool
def parse_headlines_agent():
    """
    Getting the news headlines from the i3investor blog page.
    """
    url = f"{BASE_URL}/web/headline/blog?type=research"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()

    html = resp.text
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

            results.append(
                f"Title: {a_tag.text.strip()} (Author: {author}, Date: {current_date}, URL: {full_url})"
            )

    paragraph = "Today's headlines:\n" + "\n".join(results) if results else "No headlines found for today."
    return {"paragraph": paragraph}


@tool
def parse_blog_content_agent(item: Dict) -> Dict:
    """Fetches and parses blog post content from url and extracts clean text and images."""
    url = item["url"]
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    content_div = soup.select_one("#blogcontent")
    if not content_div:
        item["content"] = ""
        return item
    paragraphs = [tag.get_text(strip=True) for tag in content_div.find_all(["h3", "p", "li"])]
    images = [img["src"] for img in content_div.find_all("img") if img.get("src")]
    item["content"] = "\n".join(paragraphs)
    item["images"] = [src.split("/")[-1] for src in images]  # Save only filenames
    return item