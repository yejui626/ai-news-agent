import json
import os
import re
from langchain_core.tools import tool

class BursaCompanyValidator:
    def __init__(self, jsonl_path: str = None):
        if jsonl_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            jsonl_path = os.path.join(base_dir, "bursa_companies.jsonl")

        self.company_map = {}
        self.companies = []
        self._load_data(jsonl_path)

    # ... [Keep your existing _normalize and _load_data methods exactly as they were] ...
    def _normalize(self, text: str) -> str:
        if not text: return ""
        text = re.sub(r'^(kl\s*[:]?\s*)', '', text.strip(), flags=re.IGNORECASE)
        return re.sub(r'[^a-zA-Z0-9]', '', text.lower())

    def _load_data(self, path):
        # ... [Paste your existing loading logic here] ...
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    self.companies.append(data)
                    if data.get("stock_code"): self.company_map[str(data["stock_code"])] = data
                    if data.get("company_short"): self.company_map[self._normalize(data["company_short"])] = data
                    for alias in data.get("aliases", []):
                        self.company_map[self._normalize(alias)] = data
            print(f"--- [Validator] Loaded {len(self.companies)} Bursa companies. ---")
        except FileNotFoundError:
            print(f"--- [Validator] Warning: Could not find {path}. ---")

    def _is_fuzzy_match(self, company_data: dict, scraped_name: str) -> bool:
        # ... [Paste your existing fuzzy match logic here] ...
        if not scraped_name: return False
        clean_scraped = self._normalize(scraped_name)
        for alias in company_data.get("aliases", []):
            clean_alias = self._normalize(alias)
            if len(clean_alias) > 3 and len(clean_scraped) > 3:
                if clean_alias in clean_scraped or clean_scraped in clean_alias:
                    return True
        return False

    def lookup(self, query: str) -> str:
        """
        Public method to be wrapped as a tool.
        """
        # Create a dummy item to reuse your logic
        item = {"name": query, "ticker": query} 
        
        # Reuse your robust validate_and_enrich logic
        result = self.validate_and_enrich(item)
        
        if result:
            return json.dumps({
                "is_listed": True,
                "official_name": result["company_official_name"],
                "stock_code": result["ticker"],
                "status": "Match Found"
            })
        else:
            return json.dumps({
                "is_listed": False,
                "status": "Not Listed"
            })

    def validate_and_enrich(self, scraped_item: dict) -> dict | None:
        # ... [Paste your existing validate_and_enrich logic here] ...
        # (This remains the logic engine for the tool)
        scraped_name = scraped_item.get("name", "")
        scraped_ticker = scraped_item.get("ticker", "")
        match = None
        clean_ticker = self._normalize(scraped_ticker)
        
        # Strategy 1 & 2 & 3 (Same as your previous valid code)
        if clean_ticker and clean_ticker in self.company_map:
            candidate = self.company_map[clean_ticker]
            if self._is_fuzzy_match(candidate, scraped_name) or not scraped_name:
                match = candidate
        
        if not match and scraped_name:
            clean_scraped = self._normalize(scraped_name)
            if clean_scraped in self.company_map:
                match = self.company_map[clean_scraped]
            if not match:
                for company in self.companies:
                    if self._is_fuzzy_match(company, scraped_name):
                        match = company
                        break
        
        if match:
            return {
                "name": scraped_item.get("name"),
                "ticker": match.get("stock_code"),
                "company_official_name": match.get("company_long")
            }
        return None

# ==========================================
#  ✅ THE TOOL DEFINITION
# ==========================================
# We instantiate the validator globally or pass it in. 
# For simplicity, we create a function that instantiates/uses it.

_validator_instance = BursaCompanyValidator()

@tool
def check_bursa_listing(company_name_or_ticker: str) -> str:
    """
    Checks if a company is listed on Bursa Malaysia.
    Input can be a company name (e.g. "Infomina") or a ticker (e.g. "INFOM" or "0265").
    Returns JSON with 'is_listed', 'official_name', and 'stock_code' if found.
    ALWAYS call this tool before extracting a company to ensure it is valid.
    """
    return _validator_instance.lookup(company_name_or_ticker)