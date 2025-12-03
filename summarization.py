import os
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Assuming costar_prompt.py is importable
from costar_prompt import CostarPrompt 


load_dotenv()

# ==============================================================================
# 1. Summarizer Class
# ==============================================================================
class Summarizer:
    """
    Handles the summarization of individual news articles using a fine-tuned LLM 
    and the CoSTAR prompting framework.
    """
    
    def __init__(self, model_id: str = "ft:gpt-4.1-nano-2025-04-14:universiti-malaya:summarizer:CNgrReGm"):
        """
        Initializes the LLM with the specified fine-tuned model ID.
        """
        self.model_id = model_id
        # Keeping temperature low for deterministic, factual summarization
        self.llm = ChatOpenAI(
            model=self.model_id,
            temperature=0.0 
        )
        
        # System Message acts as the high-level instruction set for the LLM's personality
        self.system_message = SystemMessage(
            content="You are a financial news summarization assistant. Your task is to read a full financial or corporate news article and generate a concise abstractive summary.\n\nYour summary must:\n- Contain 3 to 4 sentences.\n- Present key corporate or financial developments factually and neutrally.\n- Use a professional tone similar to financial journalism.\n- Avoid opinions, redundant details, or speculative language.\n- Use proper capitalization and standard financial abbreviations (e.g., RM, %, bn, m)."
        )

    def summarize(self, article_content: str) -> str:
        """
        Generates a summary using the CoSTAR prompt structure.
        """
        
        # 1. Define the CoSTAR components
        costar = CostarPrompt(
            context="You are a corporate news analyst preparing brief updates for an investment report.",
            objective="Analyze the full corporate news article and generate a concise abstractive summary.",
            audience="The target audience is a senior investment editor or portfolio manager who needs quick, factual insights.",
            response="Generate a summary that: 1) Is 3 to 4 sentences long. 2) Presents key corporate/financial developments factually. 3) Maintains a neutral, professional financial journalism tone. 4) Avoids opinions, redundancy, or speculative language. 5) Uses proper capitalization and standard financial abbreviations (RM, %, bn, m)."
        )
        
        # 2. Combine the CoSTAR prompt structure with the article content
        full_prompt_content = f"{str(costar)}\n\n### SOURCE ARTICLE ###\n{article_content}"

        messages = [
            self.system_message,
            HumanMessage(content=full_prompt_content)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            print(f"--- [Summarizer] Error summarizing article: {e} ---")
            return "[Summarization Failed]"

    def process_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes a list of articles (from scraper) and adds a 'summary' key to each.
        """
        processed_articles = []
        for article in articles:
            article_content = article.get("text", "")
            
            if article_content and article_content != "[no content extracted]":
                article["summary"] = self.summarize(article_content)
            else:
                article["summary"] = "[No article text available for summarization]"
                
            processed_articles.append(article)
            
        return processed_articles


# ==============================================================================
# 2. LangGraph Node Wrapper
# ==============================================================================
def summarizer_node(state: dict) -> dict:
    """
    LangGraph Node function to summarize all scraped articles.
    """
    articles = state.get("scraped_articles", [])
    
    if not articles:
        print("--- [Summarizer Node] No articles found in state. Skipping. ---")
        return {"scraped_articles": []}
    
    print(f"--- [Summarizer Node] Starting summarization for {len(articles)} articles ---")
    
    summarizer = Summarizer()
    
    processed_articles = summarizer.process_articles(articles)
    
    print("--- [Summarizer Node] Summarization complete. ---")
    
    return {"scraped_articles": processed_articles}


# ==============================================================================
# 3. Example Output (Based on Fine-tuned Model)
# ==============================================================================
if __name__ == "__main__":
    example_text = """KUALA LUMPUR (Oct 6): Lion Industries Corp Bhd (KL:LIONIND) plans to revitalise its steel business by, among others, seeking strategic partners for two of its Amsteel Mills plants to bring in new processes amid challenging times.

“We wish to stress that the steel business is still very much the core business of Lion Industries Corporation Bhd and have no plans to shut Amsteel Mills other than for upgrading purposes,” it said in a filing with Bursa Malaysia in response to an article published in The Edge Weekly (Oct 6-12, 2025 issue).

Lion Industries said it is planning to install new machinery and processes at the Bukit Raja mill, which has been in operation since 1978, to enhance efficiency and competitiveness. The group is also in talks with potential strategic partners to provide new technologies.“This will make the plant more efficient and cost-competitive, and keep abreast with market conditions and demands,” it said.

As for the Banting plant, the group said it has been temporarily idled due to high operating costs, particularly following the hike in electricity tariffs that rendered operations uneconomical.

To revitalise operations, the group is exploring strategic partnerships to introduce new processes and products for the plant. An announcement will be made once a suitable partner has been secured, it added.

Lion Industries is principally involved in steel, property development, building materials and others.

The group’s share price closed at a two-month low of 17.5 sen — down 10.26% or two sen — on Monday, valuing it at RM122.49 million. Year to date, the stock is down 23.91%."""

    print("--- Running Summarizer Standalone ---")
    summarizer = Summarizer()
    summary = summarizer.summarize(example_text)
    
    print("\n✅ GENERATED SUMMARY (3-4 sentences):")
    print(summary)