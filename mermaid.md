```mermaid
graph TD
 D["🧠 GenerateAnswerNode<br/>(LLM Agent)"]
 E["🔍 Tool: check_bursa_listing"]
 F["🗄️ Ground Truth<br/>bursa_companies.jsonl"]

 D --> E
 E --> F

 click D "https://python.langchain.com/docs/langgraph/" "LangGraph Agent Pattern"
 click E "https://python.langchain.com/docs/integrations/tools/" "Tool Invocation Reference"
 click F "https://www.bursamalaysia.com/market_information/equities_prices" "Bursa Malaysia Reference"