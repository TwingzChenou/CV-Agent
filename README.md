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

*   **🧠 Static Knowledge (RAG):** Uses **Pinecone** to store vector embeddings of my Curriculum Vitae (`cv.json`). This allows the agent to answer questions about my education, past experience, and skills with high precision.
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
    Create a `.env` file in the root directory and add the following keys:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_INDEX=your_index_name
    GITHUB_TOKEN=your_github_personal_access_token
    ```

3.  **Ingest CV Data:**
    Upload your CV data to Pinecone.
    ```bash
    python loader.py
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
│   │   └── main.py         # App Entrypoint
│   ├── evaluation/         # Evaluation Responses Gemini
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
*   **Intent Classification:** Uses **DSPy** to classify user queries into categories like `chitchat` (handled directly), `cv` (requires RAG), or `list_all_projects` (requires GitHub API).
*   **System Prompt:** Defines the persona of the agent ("Quentin Forget") and sets strict behavioral rules (STAR method, professional tone).
*   **Agent Initialization:** Configures the **LlamaIndex ReActAgent** with the necessary tools and LLM (Gemini 2.5 Flash).
*   **Response Generation:** Route the query to either a direct LLM call or the Agent based on the classified intent.

### `tools.py`
**The Toolbelt.** This file defines the specific capabilities (tools) the agent can use.
*   **GitHub Integration:**
    *   `list_github_projects`: Fetches the user's public repositories using the GitHub API.
    *   `get_github_activity`: Retrieves the README content of a specific repository for real-time project context.
*   **CV RAG Tool:**
    *   `cv_query_engine`: Creates a query engine connected to the Pinecone vector database to answer questions about the CV.
*   **Tool Assembly:** The `get_tools()` function packages these functions into LlamaIndex-compatible `FunctionTool` objects for the agent.

### `index.py`
**The Vector Manager.** This file manages the Pinecone vector database connection and indexing.
*   **Connection:** Establishes the connection to the Pinecone index using environment variables.
*   **Embedding Model:** Configures `GeminiEmbedding` (text-embedding-004) to convert text into vector representations.
*   **Indexing Pipeline:** Defines the `run_indexing_pipeline` function which takes document chunks, generates embeddings, and upserts them into Pinecone.

### `loader.py`
**The Data Ingestor.** This file handles the ETL (Extract, Transform, Load) process for the CV data.
*   **Loading:** Uses `SimpleDirectoryReader` to load the PDF resume (`CV_Quentin_Forget.pdf`) from the `data/` directory.
*   **Splitting:** Uses `SentenceSplitter` to break the document into manageable chunks (tokens) with overlap to preserve context.
*   **Execution:** Calls the indexing pipeline from `index.py` to store the processed chunks in the vector database.

## 🧪 Evaluation Framework Details

The `backend/evaluation` directory contains scripts to ensure the quality and accuracy of the agent's responses using the **Ragas** framework.

### `generate_dataset.py`
**The Scenario Generator.** This script automates the creation of test cases.
*   **Tool Analysis:** Iterates through available tools in `tools.py`.
*   **Scenario Creation:** Uses Gemini to invent 5 distinct user questions per tool that *must* use that specific tool to be answered correctly.
*   **Output:** Generates a JSON dataset (`agent_dataset.json`) of test queries.

### `run_eval.py`
**The Judge.** This script runs the evaluation pipeline to measure performance.
*   **Ragas Integration:** Uses Ragas (with Gemini as the judge LLM) to compute metrics like **Faithfulness** (is the answer derived from context?) and **Answer Relevancy** (does it answer the question?).
*   **Batch Inference:** Runs the agent against a dataset of questions (currently a mix of hardcoded golden datasets and generated ones).
*   **Reporting:** Outputs a pandas DataFrame and CSV with scores for each question, helping identify weak points in the system prompt or retrieval logic.

## Key Features

*   **Hybrid Search:** Combines keyword interactions with semantic understanding to retrieve the most relevant information from the CV.
*   **Real-time Tooling:** Implementation of the ReAct pattern allows the agent to autonomously decide when to query the GitHub API for live data versus when to rely on internal knowledge.
*   **Eval-Driven Development:** Quality of responses is monitored using evaluation frameworks (like Ragas/Custom scripts) to ensure accuracy and relevance.
