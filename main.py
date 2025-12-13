import operator
import asyncio
import nest_asyncio
from typing import Annotated, List, TypedDict, Dict, Any
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send

# Apply nest_asyncio for Jupyter/Async environments
nest_asyncio.apply()

# --- Import Capabilities ---
from scraper import TheEdgeScraper
from summarization import Summarizer
from translation import Translator

# --- Report Generator (Synthesizer Logic) ---
def generate_markdown_report(articles: list, filename: str = "Market_Watch_Report.md"):
    """Compiles the final report from processed articles."""
    # Sort by original index to keep order, or by ticker
    articles.sort(key=lambda x: int(x.get('id', 0)))
    
    content = f"# 📈 AI Market Watch Report\n**Generated via Orchestrator-Worker Architecture**\n\n---\n\n"
    
    for art in articles:
        title = art.get('title', 'Untitled')
        t_title = art.get('title_translated', title)
        summary = art.get('summary', 'No summary.')
        t_summary = art.get('summary_translated', summary)
        url = art.get('url', '#')
        
        content += f"## {title}\n"
        content += f"**🇨🇳 {t_title}**\n\n"
        content += f"**Summary:** {summary}\n\n"
        content += f"**摘要:** {t_summary}\n\n"
        content += f"[Read Source]({url})\n\n---\n"
        
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"📄 Report saved to {filename}")

# ====================================================
# 1. Standalone Scraper Process
# ====================================================

def run_scraper_process(url: str) -> List[Dict]:
    """
    Independent process to fetch data.
    This is NOT part of the LangGraph workflow.
    """
    print(f"\n📡 [System] Starting Scraper Process for {url}...")
    scraper = TheEdgeScraper()
    raw_data = scraper.run(url=url)
    
    # Pre-processing: Add IDs for tracking
    clean_data = []
    for idx, item in enumerate(raw_data):
        clean_data.append({
            "id": idx, # Simple integer ID for sorting later
            "url": item.get('url'),
            "title": item.get('title'),
            "content": item.get('text', '') or item.get('summary', ''),
            "summary": None # Placeholder
        })
        
    print(f"✅ [System] Scraper finished. Passed {len(clean_data)} articles to pipeline.\n")
    return clean_data

# ====================================================
# 2. Orchestrator-Worker Graph Definitions
# ====================================================

# Global State (The "Job" State)
class JobState(TypedDict):
    raw_input: List[Dict]  # The input from the scraper
    valid_tasks: List[Dict] # Filtered list ready for assignment
    processed_results: Annotated[List[Dict], operator.add] # Reducer list
    final_report: str

# Worker State (The "Task" State)
class TaskState(TypedDict):
    article: Dict

# --- Node: The Orchestrator ---
def orchestrator_node(state: JobState):
    """
    The Planning Node.
    Analyzes input, filters invalid data, and prepares the task list.
    """
    raw = state.get("raw_input", [])
    print(f"🧠 [Orchestrator] Planning tasks for {len(raw)} items...")
    
    # Example Logic: Filter out articles with no content
    valid_tasks = [a for a in raw if len(a.get("content", "")) > 50]
    
    ignored = len(raw) - len(valid_tasks)
    if ignored > 0:
        print(f"   ⚠️ Ignoring {ignored} articles (too short/empty).")
        
    return {"valid_tasks": valid_tasks}

# --- Edge Logic: The Delegate ---
def map_tasks(state: JobState):
    """
    Maps the valid tasks to the Worker Node in parallel.
    """
    tasks = state.get("valid_tasks", [])
    
    # Dynamic Fan-Out
    return [Send("worker_node", {"article": t}) for t in tasks]

# --- Node: The Worker ---
def worker_node(state: TaskState):
    """
    The Execution Node.
    Performs Summarization -> Translation.
    """
    article = state["article"]
    print(f"   ⚙️ [Worker #{article['id']}] Processing: {article['title'][:20]}...")
    
    # 1. Summarize
    summarizer = Summarizer()
    # Using agentic=True for quality, False for speed
    summary = asyncio.run(summarizer.run_agentic_summarization(article['content']))
    article['summary'] = summary
    
    # 2. Translate (Only if summary exists)
    if summary:
        translator = Translator()
        
        t_title = translator.run_agentic_translation(
            article['title'], target_lang="Simplified Chinese (Malaysia)"
        )
        article['title_translated'] = t_title.get('current_draft')
        
        t_sum = translator.run_agentic_translation(
            summary, target_lang="Simplified Chinese (Malaysia)"
        )
        article['summary_translated'] = t_sum.get('current_draft')
        
    return {"processed_results": [article]}

# --- Node: The Synthesizer ---
def synthesizer_node(state: JobState):
    """
    The Reducer Node.
    Compiles results.
    """
    print("✨ [Synthesizer] All tasks complete. Generating report...")
    results = state.get("processed_results", [])
    
    output_path = "Market_Watch_Report.md"
    generate_markdown_report(results, filename=output_path)
    
    return {"final_report": output_path}

# ====================================================
# 3. Build & Run
# ====================================================

def build_pipeline():
    workflow = StateGraph(JobState)
    
    # Add Nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("worker_node", worker_node)
    workflow.add_node("synthesizer", synthesizer_node)
    
    # Define Flow
    workflow.add_edge(START, "orchestrator")
    
    # Orchestrator -> Workers (Fan-Out)
    workflow.add_conditional_edges(
        "orchestrator",
        map_tasks,
        ["worker_node"]
    )
    
    # Workers -> Synthesizer (Fan-In)
    workflow.add_edge("worker_node", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()

if __name__ == "__main__":
    # 1. Run the separate Scraper Process
    target_url = "https://theedgemalaysia.com/categories/Corporate"
    scraped_data = run_scraper_process(target_url)
    
    if scraped_data:
        # 2. Initialize the O-W Workflow with the data
        print("🚀 [System] Handing over data to Orchestrator-Worker Pipeline...")
        app = build_pipeline()
        
        # Inject data into the graph state
        initial_state = {
            "raw_input": scraped_data,
            "processed_results": [] # Initialize reducer
        }
        
        asyncio.run(app.ainvoke(initial_state))
        print("✅ System Finished Successfully.")
    else:
        print("❌ Scraper returned no data. Aborting pipeline.")