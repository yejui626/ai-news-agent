# ai-news-agent

Below is a formal, corporate‑style statement of scope for your “News‑to‑Pitch” portfolio project, aligned with our revised architecture and your requirements.  Each item references the relevant section of our prior design for clarity.

---

## Project Scope

### 1. Automated Ingestion & Deck Generation (Backend Scheduler)

* **Daily News Scraping**

  * Crawl and parse target URLs/RSS feeds (e.g. KLSE i3Investor) each trading day after market close.
  * Extract “Macro Review & Outlook” sections and individual “Company News‑Bite” items (per Phase 2: News Scraping & Data Extraction).
* **Content Summarization Agents**

  * Invoke GenAI summarization for macro and per‑company content (per Phase 3: GenAI Content Generation).
  * Apply translation or validation agents as configured.
* **Embedding & Indexing**

  * Generate embeddings for each section.
  * Upsert into a local Chroma vector store to support semantic retrieval (per “Revised End‑to‑End Workflow”).
* **PPTX Assembly**

  * Assemble slides using `python‑pptx`: one macro slide + one slide per company (per Slide Generation Pipeline).
  * Persist PPTX files in storage and record metadata in SQLite (`pptx_records` table).
* **Scheduling & Monitoring**

  * Implement a reliable scheduler (e.g. APScheduler) within the FastAPI service.
  * Add basic success/failure logging for audit and alerting.

*References: Revised End‑to‑End Workflow (Backend Scheduler / Ingestion Service)*

---

### 2. Persistence Layer

* **SQLite Database**

  * Tables: `users`, `conversations`, `transcripts`, `pptx_records` (per Persistence Layer design).
  * Store deck metadata, chat transcripts, and user/session information.
* **Local Vector Store (Chroma)**

  * Section‑level embedding index tagged with `date`, `section_type`, `ticker`.
  * Provides top‑K semantic retrieval for chat queries.

*References: Revised End‑to‑End Workflow (Persistence Layer)*

---

### 3. Chat API (FastAPI)\*\*

* **Endpoints**

  * `GET /decks/latest` & `GET /decks/{date}`: retrieve pre‑generated PPTX.
  * `POST /chat`:

    * Load last N turns for context carry‑over.
    * Perform semantic retrieval from Chroma.
    * Call OpenAI ChatCompletion with history + retrieved snippets.
    * Persist both user message and assistant reply.
  * `GET /history?user_id=…`: return full chat transcript.
* **Multi‑User & Session Management**

  * Support concurrent users with isolated conversation records.
  * Secure user authentication (e.g. API key or simple token).

*References: Revised End‑to‑End Workflow (FastAPI Chat & Asset API)*

---

### 4. Frontend Interface (Streamlit)\*\*

* **Deck Download Section**

  * Automatically display and allow download of the latest deck.
  * Date picker for historical decks.
* **Chat Widget**

  * Text‑input and message stream display.
  * Fetches via `POST /chat` and displays responses in real time.
  * “Clear Conversation” button to start a new session.
* **Transcript Viewer**

  * Optional panel to view/save past chat history (`GET /history`).

*References: Revised End‑to‑End Workflow (Streamlit Frontend)*

---

### 5. Non‑Functional Requirements

* **Scalability**: Able to handle daily ingestion and chat requests from multiple simultaneous users.
* **Reliability**: Automated scheduler with retry logic; logged success/failure metrics.
* **Maintainability**: Modular codebase (backend and frontend separated); CI/CD with GitHub Actions.
* **Security & Compliance**:

  * Local-only storage for portfolio demo; no PII collected.
  * API authentication for chat endpoints.

---

### 6. Deliverables & Milestones

| Milestone                                 | Deliverable                                                                 |
| ----------------------------------------- | --------------------------------------------------------------------------- |
| **M1: Project Setup & Schema Definition** | Monorepo with `/backend` and `/frontend`; SQLite models and Chroma setup.   |
| **M2: Ingestion & Summarization POC**     | Backend scheduler stub that scrapes one sample date and logs summaries.     |
| **M3: PPTX Generation**                   | Fully automated daily deck generation and storage; `GET /decks/latest`.     |
| **M4: Chat API with RAG**                 | `POST /chat` endpoint with context carry‑over and Chroma retrieval.         |
| **M5: Streamlit Frontend**                | UI for deck download and chat; integration with backend endpoints.          |
| **M6: Multi‑User & Deployment**           | Dockerized services; instructions for local deployment; demo documentation. |

---

This scope captures all functional components, non‑functional requirements, and milestone deliverables needed to launch your portfolio project.  Let me know if you’d like to adjust any of these items or dive into the first milestone immediately.
