import os
import sys
import operator
from typing import List, Dict, Any, TypedDict, Annotated
from dotenv import load_dotenv

# LangChain / LangGraph Imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END, START

# Local Imports handling (supports running from root or backend dir)
try:
    from costar_prompt import CostarPrompt
    from evaluation_service import EvaluationAgent
except ImportError:
    # Fallback: add parent directory to path
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from costar_prompt import CostarPrompt
    from evaluation_service import EvaluationAgent

load_dotenv()

# ====================================================
# 1. Define Agent State
# ====================================================
class SummarizationState(TypedDict):
    source_text: str       # The full original article content
    initial_summary: str
    current_draft: str     # The draft being evaluated/refined
    
    # Evaluation Data
    eval_score: float
    eval_feedback: str     # Feedback from the Judge
    is_passing: bool
    
    # Loop control
    retry_count: int

# ==============================================================================
# 2. Summarizer Class (The Agent)
# ==============================================================================
class Summarizer:
    """
    Handles the summarization of news articles using a fine-tuned LLM.
    Now includes a Self-Correcting Agentic Workflow (Evaluator-Optimizer).
    """
    
    def __init__(self, model_id: str = "ft:gpt-4.1-nano-2025-04-14:universiti-malaya:summarizer-07122025:Ck3tnFjB"):
        """
        Initializes the LLM and the Agent Graph.
        """
        self.model_id = model_id
        # Keeping temperature low for deterministic, factual summarization
        self.llm = ChatOpenAI(
            model=self.model_id,
            temperature=0.0 
        )
        
        # System Message for the base generator
        self.system_message = SystemMessage(
            content="You are a specialized financial summarization engine. Read the structured request and generate the summary strictly following the rules outlined in the 'Response' section of the CoSTAR prompt."
        )

        # Initialize the Judge (Evaluator)
        # We accept a slightly lower threshold for automated passing to avoid infinite loops on subjective style
        self.evaluator = EvaluationAgent(
            task_type="summarization", 
            threshold=0.85,
            model_name="gpt-5-mini" 
        )

        # Build the Graph once during initialization
        self.app = self._build_agent_graph()

    # -----------------------------------------------------------
    # Core Methods (Tools/Actions)
    # -----------------------------------------------------------
    def summarize(self, article_content: str) -> str:
        """
        Standard 0-shot summarization using CoSTAR.
        """
        costar = CostarPrompt(
            context="You are a corporate news analyst preparing brief updates for an investment report.",
            objective="Analyze the full corporate news article and generate a concise, data-heavy summary.",
            audience="The target audience is a senior investment editor or portfolio manager who needs quick, factual insights.",
            
            # ✅ IMPROVED RESPONSE INSTRUCTIONS
            response="""Generate a summary strictly adhering to these rules:
            1. **Structure:** Exactly 3 to 5 sentences.
            2. **Content:** Prioritize hard data (Revenue, Profit, Dates).
            3. **Tone:** Professional financial journalism.
            4. **Quote Policy:** - **PERMITTED:** You MAY use quotes for key strategic announcements (e.g., "Company searching for new CEO", "Targeting completion by 2026").
            - **FORBIDDEN:** DO NOT use quotes for emotional/promotional language (e.g., "We are delighted," "We are proud").
            5. **Formatting:** Use standard abbreviations (RM, %, bn, m)."""
        )
        
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

    def refine_summary(self, source_text: str, current_draft: str, improvements: str) -> str:
        """
        Refines the summary based on external evaluation feedback.
        """
        refinement_costar = CostarPrompt(
            context="You are a meticulous Senior Editor, tasked with refining a corporate news summary based on critical feedback. Your primary goal is to fix factual errors and ensure the tone is highly professional.",
            objective=f"Rework the summary to strictly address the following feedback and constraints:\n\n### FEEDBACK ###\n{improvements}",
            audience="The final audience is a senior investment editor.",
            response=f"Output the final refined summary (3-5 sentences ONLY) that is factually flawless and maintains an impeccable executive tone. Base all facts ONLY on the source article provided below.\n\n### SOURCE ARTICLE ###\n{source_text}"
        )
        
        prompt_content = f"{str(refinement_costar)}\n\n### CURRENT DRAFT ###\n{current_draft}"

        messages = [
            self.system_message,
            HumanMessage(content=prompt_content)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            return current_draft # Fallback to previous draft if refinement fails

    # -----------------------------------------------------------
    # Graph Nodes (Internal)
    # -----------------------------------------------------------
    def _initial_summarization_node(self, state: SummarizationState) -> Dict[str, Any]:
        print(f"\n--- [Step 1] Initial Summarization ---")
        res = self.summarize(state["source_text"])
        return {
            "initial_summary": res, 
            "current_draft": res, 
            "eval_feedback": "",
            "retry_count": 0
        }

    async def _evaluation_node(self, state: SummarizationState) -> Dict[str, Any]:
        print(f"--- [Step 2] Evaluation (DeepEval) ---")
        
        # Async evaluation via the EvaluationService
        results = await self.evaluator.a_evaluate(
            generated_text=state["current_draft"],
            source_context=state["source_text"],
            section_topic="Corporate News Summary"
        )
        
        feedback_list = []
        for name, m in results["metrics"].items():
            if not m["passed"]:
                feedback_list.append(f"[{name}] Score: {m['score']:.2f}. Feedback: {m['feedback']}")
        
        feedback_str = " ".join(feedback_list)
        if not feedback_str and not results["overall_pass"]:
             # Fallback if metrics failed but didn't provide clear feedback string
             feedback_str = "General failure in tone or factual fidelity."

        return {
            "eval_score": results["average_score"],
            "is_passing": results["overall_pass"],
            "eval_feedback": feedback_str
        }

    def _refinement_node(self, state: SummarizationState) -> Dict[str, Any]:
        current_retries = state["retry_count"]
        print(f"--- [Step 3] Refinement (Attempt {current_retries + 1}) ---")
        
        refined = self.refine_summary(
            source_text=state["source_text"],
            current_draft=state["current_draft"],
            improvements=state["eval_feedback"]
        )
        
        return {
            "current_draft": refined, 
            "retry_count": current_retries + 1
        }

    def _check_quality(self, state: SummarizationState) -> str:
        if state["is_passing"]:
            return "pass"
        
        if state["retry_count"] >= 3:
            print("   🛑 Max retries reached. Returning best effort.")
            return "pass" 
            
        return "retry"

    # -----------------------------------------------------------
    # Graph Construction
    # -----------------------------------------------------------
    def _build_agent_graph(self):
        workflow = StateGraph(SummarizationState)
        
        # Nodes
        workflow.add_node("initial", self._initial_summarization_node)
        workflow.add_node("evaluate", self._evaluation_node)
        workflow.add_node("refiner", self._refinement_node)
        
        # Edges
        workflow.add_edge(START, "initial")
        workflow.add_edge("initial", "evaluate")
        
        workflow.add_conditional_edges(
            "evaluate",
            self._check_quality,
            {"pass": END, "retry": "refiner"}
        )
        
        workflow.add_edge("refiner", "evaluate")
        
        return workflow.compile()

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------
    async def run_agentic_summarization(self, text: str):
        """
        Executes the self-correcting summarization workflow (Async).
        """
        inputs = {
            "source_text": text,
            "initial_summary": "",
            "current_draft": "",
            "eval_score": 0.0,
            "eval_feedback": "",
            "is_passing": False,
            "retry_count": 0
        }
        
        print(f"🚀 Starting Summarization Agent...")
        final_state = await self.app.ainvoke(inputs)
        return final_state["current_draft"]

    def process_articles(self, articles: List[Dict[str, Any]], use_agentic: bool = False) -> List[Dict[str, Any]]:
        """
        Takes a list of articles and adds a 'summary' key.
        Supports switching between fast (standard) and robust (agentic) modes.
        """
        import asyncio # Import here to avoid top-level async issues in some envs
        
        processed_articles = []
        for article in articles:
            content = article.get("text", "")
            
            if content and content != "[no content extracted]":
                if use_agentic:
                    # Run the agent loop (requires async context handling if called synchronously)
                    try:
                        summary = asyncio.run(self.run_agentic_summarization(content))
                    except RuntimeError:
                        # Fallback for nested event loops (e.g. inside Jupyter)
                        import nest_asyncio
                        nest_asyncio.apply()
                        summary = asyncio.run(self.run_agentic_summarization(content))
                else:
                    summary = self.summarize(content)
            else:
                summary = "[No article text available]"
                
            article["summary"] = summary
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
        return {"scraped_articles": []}
    
    print(f"--- [Summarizer Node] Starting processing for {len(articles)} articles ---")
    
    summarizer = Summarizer()
    
    # We use agentic=True for high quality, or False for speed. 
    # For now, let's default to False (Fast) unless specified in state, 
    # as Agentic takes ~3x longer per article.
    use_agentic = state.get("use_agentic_summarizer", False) 
    
    processed_articles = summarizer.process_articles(articles, use_agentic=use_agentic)
    
    return {"scraped_articles": processed_articles}

# ==============================================================================
# 3. Execution Test
# ==============================================================================
if __name__ == "__main__":
    import asyncio
    
    # Test Data
    example_text = """KUALA LUMPUR (Oct 6): Lion Industries Corp Bhd (KL:LIONIND) plans to revitalise its steel business... [Truncated for brevity] ..."""

    summarizer = Summarizer()
    
    print("--- 1. Testing Standard Summarization ---")
    print(summarizer.summarize(example_text))
    
    print("\n--- 2. Testing Agentic Summarization ---")
    final_sum = asyncio.run(summarizer.run_agentic_summarization(example_text))
    print(f"Final Agentic Summary: {final_sum}")