import json
import pandas as pd
from datetime import date, timedelta
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from company_validator import check_bursa_listing
# Import the graph engine
from scrapegraphai.graphs import SmartScraperMultiGraph
from extract_original_news import extract_from_results
from company_validator import BursaCompanyValidator

# Load environment variables (API keys)
load_dotenv()

# ==============================================================================
# 1. Pydantic Schemas (Data Structure)
# ==============================================================================
class Company(BaseModel):
    """Extracted data about company's news."""
    name: Optional[str] = Field(default=None, description="Name of the company")
    ticker: Optional[str] = Field(default=None, description="Ticker symbol of the company")
    title: Optional[str] = Field(default=None, description="Title of the news article")
    description: List[str] = Field(default=None, description="News content description below the title")
    link: Optional[str] = Field(default=None, description="Link to the news article")

class News(BaseModel):
    """Extracted data about news."""
    company: List[Company] = Field(
        default=None, description="Extracted data about companies' news."
    )

# ==============================================================================
# 2. Scraper Class
# ==============================================================================
class TheEdgeScraper:
    def __init__(self, model_name: str = "openai/gpt-4o-mini", headless: bool = True):
        self.config = {
            "llm": {"model": model_name},
            "headless": headless,
            "verbose": True,
            "loader_kwargs": {
                "click_selectors": [
                    "div.LoadMoreButton_btnWrapper__CtkKX",
                    "div.LoadMoreButton_btnWrapper__CtkKX > span",
                    "text='Load More'"
                ],
                "max_clicks": 5,
                "item_selector": "div.NewsList_newsListItemWrap__XovMP",
                "wait_after_click": 1000, 
            },
            "tools": [check_bursa_listing]
        }
        self.base_url = "https://theedgemalaysia.com"
        self.validator = BursaCompanyValidator()

    def _build_prompt(self, link: str) -> str:
        # Calculate today and yesterday to handle timezone/midnight edge cases
        today = date.today()
        yesterday = today - timedelta(days=1)
        date_str = f"{today.strftime('%d %b %Y')} or {yesterday.strftime('%d %b %Y')}"
        
        return f"""
        You are a website scraper. Extract corporate news related to Malaysia/Bursa from the link: {link}.
        
        **Date Criteria:**
        Only extract news published on **{date_str}**. 
        If the date is not explicitly visible, assume news at the top of the list is recent.

        **Relevance Criteria:**
        - Must be published by theedgemalaysia.com.
        - Must be about companies listed in Bursa Malaysia.
        - IGNORE news about non-Bursa companies or international politics.

        **Extraction Rules:**
        - Extract the full title.
        - Extract the relative link (e.g. /node/...) or full link.
        - Extract the ticker if mentioned (e.g. KL:MAYBANK), else leave null.
        - Extract only if the ticker/name is validated from the `check_bursa_listing` tool.
        """

    def run(self, url: str = "https://theedgemalaysia.com/categories/Corporate", save_excel: bool = False) -> List[Dict[str, Any]]:
        print(f"--- [Scraper] Starting Scrape on {date.today()} ---")

        # 1. Initialize Graph
        smart_scraper_multi_graph = SmartScraperMultiGraph(
            prompt=self._build_prompt(url),
            source=[url],
            config=self.config,
            schema=News,
        )

        # 2. Run Graph (Scrape Headlines/Links)
        try:
            raw_result = smart_scraper_multi_graph.run()
            
            # --- FIX 1: Handle String Output (e.g. "No answer found.") ---
            if isinstance(raw_result, str):
                if "No answer found" in raw_result or "no info" in raw_result.lower():
                    print(f"--- [Scraper] Info: Agent returned '{raw_result}'. Likely no news matching date filters.")
                    return []
                
                print(f"--- [Scraper] Warning: Received string output. Attempting to parse JSON... ---")
                try:
                    # Clean up code blocks if present
                    if "```json" in raw_result:
                        import re
                        match = re.search(r"```json\s*(\{.*?\})\s*```", raw_result, re.DOTALL)
                        if match:
                            raw_result = json.loads(match.group(1))
                        else:
                            raw_result = json.loads(raw_result)
                    else:
                         raw_result = json.loads(raw_result)
                except Exception:
                    print(f"--- [Scraper] Failed to parse string output. Returning empty list. ---")
                    return []
            # -----------------------------------------------------------

            # Basic validation
            items_found = raw_result.get('company', [])
            if not items_found:
                print("--- [Scraper] No items found matching criteria. ---")
                return []
            
            # ============================================================
            # ✅ NEW: Validation & Filtering Phase
            # ============================================================
            final_articles_for_extraction = []
            for item in items_found:
                link = item.get("link", "")
                
                # Check if link exists and starts with / (is relative)
                if link and isinstance(link, str) and link.startswith("/"):
                    item["link"] = f"{self.base_url}{link}"
                
                # Ensure 'url' field also exists for extract_from_results compatibility
                # This ensures the extraction logic finds the link
                item["url"] = item["link"] 
                
                final_articles_for_extraction.append(item)
                
            raw_result['company'] = final_articles_for_extraction
            print(f"--- [Scraper] Phase 1 Complete. {len(final_articles_for_extraction)} valid Bursa articles found. ---")
            # ============================================================
            
        except Exception as e:
            print(f"--- [Scraper] Error during graph execution: {e} ---")
            import traceback
            traceback.print_exc()
            return []

        # 3. Extract Full Content
        try:
            # extract_from_results handles the 'newspaper3k' / 'BS4' extraction
            # We pass the cleaned raw_result
            df = extract_from_results(raw_result, output_xlsx="scraper_output_temp.xlsx" if save_excel else None)
            
            # Convert DataFrame to list of dicts for Agent consumption
            articles = df.where(pd.notnull(df), None).to_dict(orient="records")

            return articles

        except Exception as e:
            print(f"--- [Scraper] Error during full content extraction: {e} ---")
            return final_articles_for_extraction # Return the metadata we have if full extraction fails

# ==============================================================================
# 3. Agent Node Wrapper (For LangGraph)
# ==============================================================================
def scraper_node(state: dict):
    """
    LangGraph Node function.
    Input State: Expected to have 'target_url' (optional).
    Output State: Updates 'scraped_articles' key.
    """
    target_url = state.get("target_url", "[https://theedgemalaysia.com/categories/Corporate](https://theedgemalaysia.com/categories/Corporate)")
    
    scraper = TheEdgeScraper()
    articles = scraper.run(url=target_url)
    
    return {"scraped_articles": articles}

# ==============================================================================
# 4. Main Execution (Testing)
# ==============================================================================
if __name__ == "__main__":
    # Test the scraper directly
    print("Running Scraper Standalone...")
    scraper = TheEdgeScraper()
    results = scraper.run(save_excel=True)
    
    if results:
        print(f"\n✅ Scrape Complete. Found {len(results)} articles.")
        for i, article in enumerate(results[:3]):
            print(f"\n[{i+1}] {article.get('title')}")
            print(f"    Link: {article.get('url')}") # extract_original_news standardizes 'link' -> 'url'
            print(f"    Snippet: {str(article.get('text'))[:100]}...")
    else:
        print("No results found.")