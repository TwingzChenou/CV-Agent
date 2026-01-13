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
    python ingest_cv.py
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
├── app/                    # FastAPI Backend Application
│   ├── api/                # API Routes
│   ├── core/               # App Configuration & Security
│   ├── services/           # Business Logic (AI Agent, GitHub Service)
│   └── main.py             # App Entrypoint
├── components/             # React/Next.js UI Components
├── frontend/               # Next.js Frontend Application
│   ├── public/             # Static Assets
│   └── src/                # Frontend Source Code
├── ingest_cv.py            # Script to ingest CV JSON into Pinecone
├── cv.json                 # Raw CV Data
├── Dockerfile              # Container Configuration
└── requirements.txt        # Backend Dependencies
```

## Key Features

*   **Hybrid Search:** Combines keyword interactions with semantic understanding to retrieve the most relevant information from the CV.
*   **Real-time Tooling:** Implementation of the ReAct pattern allows the agent to autonomously decide when to query the GitHub API for live data versus when to rely on internal knowledge.
*   **Eval-Driven Development:** Quality of responses is monitored using evaluation frameworks (like Ragas/Custom scripts) to ensure accuracy and relevance.
