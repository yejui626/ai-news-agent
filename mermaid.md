graph TD
    %% ---------- Styles ----------
    classDef source fill:#e3f2fd,stroke:#0d47a1,stroke-width:2px;
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef agent fill:#fff3e0,stroke:#e65100,stroke-width:2px,stroke-dasharray: 5 5;
    classDef tool fill:#f1f8e9,stroke:#33691e,stroke-width:2px;
    classDef data fill:#ede7f6,stroke:#311b92,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef discard fill:#ffebee,stroke:#b71c1c,stroke-width:2px,stroke-dasharray: 3 3;

    %% ---------- External Source ----------
    subgraph Web_Source["External Data Layer"]
        A["🌐 The Edge Malaysia<br/>Corporate News"]:::source
    end

    %% ---------- Extraction Layer ----------
    subgraph ScrapeGraphAI["Extraction & Ingestion Layer (ScrapeGraphAI)"]
        B["🧭 Headless Browser<br/>(Playwright)"]:::process
        C["📑 ParseNode<br/>(DOM → Text)"]:::process
        D["🧠 GenerateAnswerNode<br/>(LLM Agent)"]:::agent
    end

    %% ---------- Validation Loop ----------
    subgraph Validation_Loop["Agentic Reasoning & Validation"]
        E["🔍 Tool: check_bursa_listing"]:::tool
        F["🗄️ Ground Truth<br/>bursa_companies.jsonl"]:::data
    end

    %% ---------- Output ----------
    subgraph Output["Verified Data Layer"]
        G["✅ Verified Corporate News Objects"]:::output
        H["🗑️ Discarded Noise<br/>(Intl / Non-Bursa)"]:::discard
    end

    %% ---------- Flow ----------
    A -->|Raw HTML| B
    B -->|DOM Tree| C
    C -->|Semantic Chunks| D

    %% ---------- Agentic Loop ----------
    D <--> |Entity Detection<br/>+ Tool Invocation| E
    E <--> |Whitelist Lookup| F

    %% ---------- Decision ----------
    D -->|Bursa-Matched| G
    D -.->|Filtered| H

    %% ---------- Interactivity ----------
    click D "https://python.langchain.com/docs/langgraph/" "View LangGraph Agent Pattern"
    click E "https://python.langchain.com/docs/integrations/tools/" "LLM Tool Invocation Reference"
    click F "https://www.bursamalaysia.com/market_information/equities_prices" "Bursa Malaysia Reference Data"
    click B "https://playwright.dev/" "Playwright Documentation"
