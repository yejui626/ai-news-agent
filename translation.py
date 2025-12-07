import operator
from typing import TypedDict, Annotated, List, Optional, Dict, Any
import pandas as pd
from dotenv import load_dotenv

# LangChain / LangGraph Imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

# Local Imports
from llm import generate_string, openai_llm
from costar_prompt import CostarPrompt
# Ensure evaluation_service is in your python path
try:
    from evaluation_service import EvaluationAgent
except ImportError:
    # Fallback if running from a different root
    from .evaluation_service import EvaluationAgent

load_dotenv()

# ====================================================
# 1. Define Agent State
# ====================================================
class IntegratedState(TypedDict):
    source_text: str
    source_lang: str
    target_lang: str
    initial_translation: str
    terminology_context: str
    current_draft: str
    eval_score: float
    eval_feedback: str
    is_passing: bool
    retry_count: int
    messages: Annotated[List[BaseMessage], operator.add]

# ====================================================
# 2. Translator Class
# ====================================================
class Translator:
    def __init__(self, show_prompt=False):
        self.show_prompt = show_prompt
        self.last_prompts = {}
        
        # Initialize Evaluator (Lazy load or init here)
        # We assume 0.85 is the passing threshold
        self.evaluator = EvaluationAgent(task_type="translation", threshold=0.85)
        
        # Initialize LLM for the internal agent (Scanning)
        self.agent_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Bind the static tool to the LLM
        self.tools = [self.terminology_lookup]
        self.llm_with_tools = self.agent_llm.bind_tools(self.tools)
        
        # Compile the graph once
        self.app = self._build_agent_graph()

    @staticmethod
    @tool
    def terminology_lookup(term: str) -> Optional[str]:
        """
        Look up official localized term from terminology.csv.
        Handles acronyms, partial matches, and full matches.
        """
        try:
            # Adjust path if necessary depending on where run.py is executed
            df = pd.read_csv("terminology.csv")
            df["en-MY_lower"] = df["en-MY"].str.lower().str.strip()
            term_lower = term.lower().strip()

            # 1. Exact match
            exact = df[df["en-MY_lower"] == term_lower]
            if not exact.empty:
                return exact.iloc[0]["zh-MY"]

            # 2. Acronym match
            acronym_match = df[df["en-MY_lower"].str.contains(f"({term_lower})", regex=False)]
            if not acronym_match.empty:
                return acronym_match.iloc[0]["zh-MY"]

            # 3. Partial fuzzy match
            partial = df[df["en-MY_lower"].str.contains(term_lower, na=False)]
            if not partial.empty:
                return partial.iloc[0]["zh-MY"]

            return None
        except Exception as e:
            return f"[Lookup error: {str(e)}]"

    # -----------------------------------------------------------
    # Core Methods (Workers)
    # -----------------------------------------------------------
    def translate(self, source_text: str, source_lang: str, target_lang: str):
        """Perform initial translation."""
        costar_prompt = CostarPrompt(
            context=f"You are a Translator for a bank, translating financial text from {source_lang} to {target_lang}.",
            objective=f"Translate the text '{source_text}' from {source_lang} to {target_lang}. Tone should match the style of a financial report.",
            audience="Your audience is the bank's investment report senior editor.",
            response=f"Output just the translation of '{source_text}' in {target_lang}, with no explanation."
        )
        llm = openai_llm(temperature=0)
        result = generate_string(llm, str(costar_prompt), {}, show_prompt=self.show_prompt, system_prompt_only=True)
        return result.content

    def refine_translation(self, source_text: str, initial_translated_text: str, improvements: str, source_lang: str, target_lang: str):
        """Refine translation based on feedback."""
        costar_prompt = CostarPrompt(
            context=f"You are a Translator for a bank, refining translations from {source_lang} to {target_lang}.",
            objective=f"Refine the translation based on feedback.\nOriginal: '{source_text}'\nDraft: '{initial_translated_text}'\nFeedback: '{improvements}'",
            audience="Bank's senior editor.",
            response=f"Output only the final refined translation in {target_lang}."
        )
        llm = openai_llm(temperature=0)
        result = generate_string(llm, str(costar_prompt), {}, show_prompt=self.show_prompt, system_prompt_only=True)
        return result.content

    # -----------------------------------------------------------
    # Graph Nodes (Internal)
    # -----------------------------------------------------------
    def _initial_pass_node(self, state: IntegratedState):
        print(f"\n--- [Step 1] Initial Translation ---")
        res = self.translate(state["source_text"], state["source_lang"], state["target_lang"])
        return {"initial_translation": res, "current_draft": res, "retry_count": 0}

    def _terminology_scan_node(self, state: IntegratedState):
        print(f"--- [Step 2] Scanning for Terminology ---")
        if not state["messages"]:
            prompt = f"""Identify financial acronyms/agencies in: "{state['source_text']}". 
            Use `terminology_lookup` for each. 
            Output ONLY a summary list: 'Original: [Localized]'."""
            messages = [SystemMessage(content=prompt)]
        else:
            messages = state["messages"]
        
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _extract_terms_node(self, state: IntegratedState):
        return {"terminology_context": state["messages"][-1].content}

    def _refinement_node(self, state: IntegratedState):
        print(f"--- [Step 3] Refinement (Attempt {state['retry_count'] + 1}) ---")
        feedback_context = ""
        if state["retry_count"] > 0:
            print(f"   ⚠️ Fixing based on feedback: {state['eval_feedback']}")
            feedback_context = f"\nPREVIOUS JUDGE FEEDBACK (MUST FIX): {state['eval_feedback']}"

        improvements = f"MANDATORY TERMINOLOGY:\n{state['terminology_context']}\n{feedback_context}"
        
        refined = self.refine_translation(
            source_text=state["source_text"],
            initial_translated_text=state["initial_translation"],
            improvements=improvements,
            source_lang=state["source_lang"],
            target_lang=state["target_lang"]
        )
        return {"current_draft": refined, "retry_count": state["retry_count"] + 1}

    def _evaluation_node(self, state: IntegratedState):
        print(f"--- [Step 4] Evaluation (DeepEval) ---")
        # Ensure we run this synchronously within the node wrapper
        # The user's snippet used .evaluate() which handles the loop internally
        try:
            results = self.evaluator.evaluate(
                generated_text=state["current_draft"],
                source_context=state["source_text"],
                section_topic=f"Localization Map: {state['terminology_context']}"
            )
            
            feedback_list = []
            for name, m in results["metrics"].items():
                if not m["passed"]:
                    feedback_list.append(f"[{name}] {m['feedback']}")
            
            return {
                "eval_score": results["average_score"],
                "is_passing": results["overall_pass"],
                "eval_feedback": " ".join(feedback_list)
            }
        except Exception as e:
            print(f"Evaluation Failed: {e}")
            return {"is_passing": False, "eval_feedback": str(e), "eval_score": 0.0}

    # -----------------------------------------------------------
    # Graph Construction
    # -----------------------------------------------------------
    def _build_agent_graph(self):
        """Builds and compiles the LangGraph."""
        workflow = StateGraph(IntegratedState)

        # Add Nodes
        workflow.add_node("initial", self._initial_pass_node)
        workflow.add_node("scanner", self._terminology_scan_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("extractor", self._extract_terms_node)
        workflow.add_node("refiner", self._refinement_node)
        workflow.add_node("judge", self._evaluation_node)

        # Define Logic
        def check_quality(state: IntegratedState):
            if state["is_passing"]: return "pass"
            if state["retry_count"] >= 3:
                print("   🛑 Max retries reached. Returning best effort.")
                return "pass"
            return "retry"

        def check_tools(state: IntegratedState):
            if state["messages"][-1].tool_calls: return "tools"
            return "done_scanning"

        # Edges
        workflow.add_edge(START, "initial")
        workflow.add_edge("initial", "scanner")
        
        workflow.add_conditional_edges("scanner", check_tools, {"tools": "tools", "done_scanning": "extractor"})
        workflow.add_edge("tools", "scanner")
        
        workflow.add_edge("extractor", "refiner")
        workflow.add_edge("refiner", "judge")
        
        workflow.add_conditional_edges("judge", check_quality, {"pass": END, "retry": "refiner"})

        return workflow.compile()

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------
    def run_agentic_translation(self, text: str, source_lang="English", target_lang="Simplified Chinese (Malaysia)"):
        """
        Executes the self-correcting translation workflow.
        """
        inputs = {
            "source_text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "messages": []
        }
        
        print(f"🚀 Starting Translation Agent for: '{text[:50]}...'")
        final_state = self.app.invoke(inputs)
        return final_state