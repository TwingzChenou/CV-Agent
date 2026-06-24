# AI-Powered Agentic CV

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6?style=flat&logo=typescript&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat&logo=fastapi&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10-000000?style=flat&logo=llamaindex&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-8E75B2?style=flat&logo=google&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000000?style=flat&logo=pinecone&logoColor=white)

> "An interactive RAG assistant capable of answering questions about my past (CV) and my real-time coding activity (GitHub Live)."

## Architecture & Concept

This project represents a "Digital Twin" or Agentic Resume. It splits knowledge into two distinct domains to provide accurate and up-to-date responses:

*   **🧠 Static Knowledge (RAG):** Uses **Pinecone** to store vector embeddings of my Curriculum Vitae (`cv.pdf`) and a file's profile. This allows the agent to answer questions about my education, past experience, and skills with high precision.
*   **⚡ Dynamic Knowledge (Tools):** Uses the **GitHub API** and real-time tools to fetch current coding activity, recent commits, and active repositories. This ensures the agent knows what I am working on *right now*.

```mermaid
graph TD
    User[User] -->|Query| NextJS[Next.js Frontend]
    NextJS -->|API Request| FastAPI[FastAPI Backend]
    FastAPI -->|Orchestrate| Agent[AI Agent (ReAct)]
    
    subgraph "Capabilities"
        direction TB
        Agent -->|Past Info| Pinecone[(Pinecone Vector DB)]
        Agent -->|Real-time Info| GitHub[GitHub API]
    end
    
    Pinecone -->|Context| Gemini{Gemini 2.5 Flash}
    GitHub -->|Data| Gemini
    Gemini -->|Response| NextJS
```

## Tech Stack

| Component | Technologies |
| :--- | :--- |
| **Frontend** | Next.js 14, TailwindCSS, Framer Motion |
| **Backend** | Python 3.11, FastAPI |
| **AI/Orchestration** | LlamaIndex, Gemini 2.5 Flash |
| **Data** | Pinecone (Vector DB), GitHub API |
| **DevOps** | Docker, Vercel/Render |

## Getting Started

Follow these instructions to set up the project locally.

### Prerequisites
*   Node.js & npm
*   Python 3.11+
*   Docker (optional, for containerized run)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ai-agentic-resume.git
    cd ai-agentic-resume
    ```

2.  **Configure Environment Variables:**
    Create a `.env` file in the `backend/` directory and add the following keys:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_INDEX=your_index_name
    GITHUB_TOKEN=your_github_personal_access_token
    LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
    ```

3.  **Ingest CV Data:**
    Upload your CV data to Pinecone.

    **Using Docker:**
    ```bash
    docker-compose run --rm backend python app/engine/loader.py
    ```

    **Local (after installing dependencies):**
    ```bash
    cd backend
    python app/engine/loader.py
    ```

### Running the Application

**Option A: Using Docker Compose**
```bash
docker-compose up --build
```

**Option B: Local Development**

*Backend:*
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
cd backend
uvicorn app.main:app --reload
```

*Frontend:*
```bash
cd frontend
npm install
npm run dev
```

## Project Structure

```ascii
/
├── backend/                # FastAPI Backend Application
│   ├── app/                # Application Code
│   │   ├── api/            # API Routes
│   │   ├── core/           # Configuration & Security
│   │   ├── data/           # Data & Loaders
│   │   ├── engine/         # RAG Engine & Tools
│   │   ├── main.py         # App Entrypoint
│   ├── evaluation/         # Evaluation Framework
│   │   ├── datasets/       # Test Datasets & Reports (CSV / JSON)
│   │   ├── evals/          # Evaluation Scripts
│   │   │   └── generate_evals_global.py # Global Eval Runner (with Langfuse & Cost Auditing)
│   │   └── experiments/    # Generation & Diagnostics
│   │       └── generate_tests_datasets.py # Pydantic + Gemini Test Suite Generator
│   ├── logs/               # Application Logs
│   ├── Dockerfile          # Container Configuration
│   └── requirements.txt    # Python Dependencies
├── frontend/               # Next.js Frontend Application
│   ├── src/                # Source Code
│   │   ├── app/            # Next.js App Router
│   │   ├── components/     # UI Components
│   │   ├── hooks/          # Custom Hooks
│   │   └── lib/            # Utility Libraries
│   ├── public/             # Static Assets
│   └── package.json        # Node Dependencies
├── check_setup.py          # Setup Utility
└── README.md               # Documentation
```

## ⚙️ Backend Engine Details

The `backend/app/engine` directory is the brain of the application, managing the AI logic, data ingestion, and external tool integrations. Here is a breakdown of each file:

### `generate.py`
**The Orchestrator.** This file handles the main chat generation logic.
*   **Dual Ingest & Fallback Pipeline:**
    1.  **Fast Path (Gemini Context Caching):** Attempts to fetch or build the context cache directly on the Gemini server for ultra-low latencies (< 2.5s) and 75% cost savings.
    2.  **Slow Path (Standard ReAct Agent):** In case caching fails, falls back to the LlamaIndex ReAct agent powered by Pinecone RAG and live tools.
*   **Streaming & Real-Time Chunks:** Implements `generate_response_stream` yielding JSON-lines status and text events (`{"type": "status"|"text", "content": "..."}`) for instant UI updates.
*   **Intent Classification:** Uses **DSPy** to classify user queries into categories like `chitchat` (handled directly), `cv` (requires RAG), or `list_all_projects` (requires GitHub API).
*   **Cost Calculation:** Calculates prompt, cache-hit, and completion costs via dynamic pricing formulas, and integrates token auditing.
*   **System Prompt:** Defines the persona of the agent ("Quentin Forget") and sets strict behavioral rules (STAR method, professional tone).

### `chat.py` [NEW]
**The API Router.** Exposes the main FastAPI endpoint (`/api/chat`).
*   **Hybrid Streaming:** Inspects the request's `stream` boolean parameter to dynamically toggle between a standard JSON `ChatResponse` and a FastAPI `StreamingResponse`. This maintains full compatibility with legacy clients and test runs.

### `tools.py`
**The Toolbelt.** This file defines the specific capabilities (tools) the agent can use.
*   **GitHub Integration:**
    *   `list_github_projects`: Fetches the user's public repositories using the GitHub API.
    *   `get_github_activity`: Retrieves the README content of a specific repository for real-time project context.
*   **Input Sanitization:** Uses a strict regex pattern (`^[a-zA-Z0-9_-]+$`) on the repo argument to prevent path traversal and command injection exploits.
*   **Exception Handling:** Captures `GithubException` (handling 404, 403, and rates) gracefully to avoid leaking server stack traces to public clients.
*   **CV RAG Tool:**
    *   `cv_query_engine`: Creates a query engine connected to the Pinecone vector database to answer questions about the CV.
*   **Tool Assembly:** The `get_tools()` function packages these functions into LlamaIndex-compatible `FunctionTool` objects for the agent.

### `index.py`
**The Vector Manager.** This file manages the Pinecone vector database connection and indexing.
*   **Connection:** Establishes the connection to the Pinecone index using environment variables.
*   **Embedding Model:** Configures `GeminiEmbedding` (text-embedding-004) to convert text into vector representations.
*   **Indexing Pipeline:** Defines the `run_indexing_pipeline` function which takes document chunks, generates embeddings, and upserts them into Pinecone.

### `loader.py`
**The Data Ingestor.** This file handles the ETL (Extract, Transform, Load) process.
*   **Loading:** Locates the source documents (e.g., `profil_quentin.md`) in the `data/` directory.
*   **Splitting & Parsing:** Uses **LlamaParse** (via `LLAMA_CLOUD_API_KEY`) to accurately parse and convert documents (including PDFs) into markdown format for better embedding quality.
*   **Execution:** Calls the indexing pipeline from `index.py` to store the processed chunks in the vector database.

### `caching.py` [NEW]
**The Cache Manager.** Manages the Gemini Context Caching lifecycle using the `google-genai` SDK.
*   **Compilation:** Bundles static knowledge (system prompt, parsed CV PDF, profile metadata, GitHub project list, and repository READMEs) into a unified context block (~5,500 tokens).
*   **Lifecycle Management:** Searches for an active cache (`cv_agent_context_cache`). If found, it extends its TTL by 1 hour (`3600s`). If expired or non-existent, it creates a new context cache, ensuring high cache hit rates (>99%).

## 🧪 Evaluation Framework Details

The `backend/evaluation` directory contains scripts to ensure the quality and accuracy of the agent's responses using **Ragas** and **LlamaIndex Evaluation** tools.

### `generate_tests_datasets.py`
**The Scenario Generator.** This script automates the creation of high-quality test cases using Pydantic validation and Gemini JSON schemas.
*   **Structured Pydantic Models:** Defines standard structures for target tool calls (`ToolCallModel`), tool arguments (`ToolArgsModel`), and test scenarios (`ScenarioModel`).
*   **Ground Truth Ingestion:** Queries the actual RAG engine and live GitHub tools to inject real context for each scenario.
*   **J.A.R.V.I.S Tone Synthesis:** Prompts Gemini to synthesize the perfect "reference" answer, ensuring the English-Butler persona and third-person rules are followed.
*   **Output:** Generates a robust JSON dataset (`datasets/agent_test_suite_100.json`) containing 100 test scenarios.

### `generate_evals_global.py`
**The Evaluation Runner & Judge.** This script executes evaluations to score the agent's performance.
*   **Multi-Turn & Agent Metrics:** Uses Ragas evaluators to score `ToolCallAccuracy`, `AgentGoalAccuracyWithReference`, `_TopicAdherenceScore`, and `ToolCallF1`.
*   **Single-Turn & RAG Metrics:** Evaluates retrieval quality using `ContextPrecision`, `ContextRecall`, `Faithfulness`, and `AnswerRelevancy`.
*   **Token & Cost Auditing:** Tracks exact input, output, and embedding tokens consumed during agent calls and evaluation, calculating total USD cost.
*   **Langfuse Integration:** Logs evaluation runs, scores, and costs directly to the Langfuse observability dashboard for analytics.
*   **Reporting:** Saves evaluation results to a CSV report (`datasets/ragas_eval_report.csv`).

## ⚡ Optimization & Security Hardening

This project incorporates advanced performance optimization and security hardening following the **principle of least privilege**:

### Gemini Context Caching
To achieve extremely low response latencies (< 2.5s) and reduce API token consumption costs by **75%**, the application caches the static context (system instruction + CV + GitHub metadata) directly on the Gemini server.
- **Cache Name:** `cv_agent_context_cache`
- **TTL (Time to Live):** 1 hour (`3600s`), automatically extended upon each cache hit.
- **Cache Hit Rate:** **> 99%** under continuous traffic, meaning only the user's new question is charged at full input rates.

### Container Security & Hardening
- **Non-Root Execution:** 
  - Backend runs under `appuser` (UID 1000).
  - Frontend runs under the default `node` user (UID 1000).
  - Neither container runs processes as `root`.
- **Immutable Codebase:** 
  - All source code files (`.py`) inside the backend container are owned by `root:root` and set to read-only for the application process. This prevents runtime code injection or file overwriting in case of Remote Code Execution (RCE) exploits.
  - The application process is only granted write access to the `/app/logs` directory.

### API & Code Sanitization
- **Strict Input Validation:** Input repository names in `get_github_activity` are validated against a strict regular expression (`^[a-zA-Z0-9_-]+$`) to prevent path traversal or parameter injection attacks.
- **Robust Exception Handling:** GitHub API calls handle rate-limiting and missing repositories gracefully with dedicated logging.

### Permissions & Secret Management
- **GitHub Actions Workflow:** CI permissions are locked down to `permissions: contents: read` to protect repository integrity.
- **Pinecone Vector Database:** FastAPI only queries embeddings. In production (e.g., Render), configure the environment using a **Read-Only API Key**. The Read-Write API Key should only be used locally for the data ingestion script (`loader.py`).
- **GitHub Token:** Use a personal access token (PAT) restricted strictly to public repository read scopes.

## 🧹 Repository Cleanup & Maintenance

To keep the codebase clean and avoid security risks, obsolete and temporary test scripts were removed:
- `backend/list_embeddings.py` (obsolete local testing script)
- `backend/test_embedding_local.py` (obsolete local embedding verification)
- `test_agent_manual.py` (legacy manual testing)
- `test_github_read.py` (legacy manual github tests)
- `backend/evaluation/run_eval_llamaIndex.py` (obsolete native LlamaIndex evaluator, replaced by Ragas runner)
- `backend/evaluation/experiments/generate_dataset.py` (legacy local Ollama dataset generator)
- `backend/evaluation/experiments/test_ollama_connection.py` (legacy local Ollama connectivity test)
- `backend/evaluation/evals/test_metrics_ragas.py` (broken/obsolete local Ragas evaluation attempt)
- `backend/evaluation/evals/test_agent_response.py` (legacy manual playground script for response formatting)
- `backend/evaluation/datasets/agent_RAG_dataset.json` (obsolete first 61-question scenario dataset)

## 🧪 Testing the Security & Caching Features

You can run the unit test suite to verify cost calculations, caching fallback logic, and strict regex validations:
```bash
PYTHONPATH=backend .venv/bin/pytest backend/tests
```

## Key Features

*   **⚡ Real-Time Streaming (SSE/NDJSON):** Response chunks are streamed word-by-word to the frontend using JSON-lines formatting, reducing perceived first-token latency to ~150ms.
*   **📡 Live Status Indicators:** The frontend displays the agent's real-time internal status (e.g., "Recherche dans le cache de contexte...", "Classification de la demande...") to provide visual feedback during processing.
*   **Hybrid Search:** Combines keyword interactions with semantic understanding to retrieve the most relevant information from the CV.
*   **Real-time Tooling:** Implementation of the ReAct pattern allows the agent to autonomously decide when to query the GitHub API for live data versus when to rely on internal knowledge.
*   **Eval-Driven Development:** Quality of responses is monitored using evaluation frameworks (like Ragas/Custom scripts) to ensure accuracy and relevance.
