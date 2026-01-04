import os
import dspy
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Setup Gemini
llm = Gemini(model="models/gemini-2.5-flash", api_key=GOOGLE_API_KEY)
Settings.llm = llm

# Setup DSPy
lm = dspy.LM("gemini/gemini-2.5-flash", api_key=GOOGLE_API_KEY)
dspy.settings.configure(lm=lm)

# --- DSPy Intent Classifier ---
class IntentSignature(dspy.Signature):
    """Classify the user query into one of the following intents: 'github', 'cv', 'chitchat', 'mixed'."""
    query = dspy.InputField()
    intent = dspy.OutputField(desc="One of: github, cv, chitchatmixed")

class IntentClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(IntentSignature)

    def forward(self, query):
        return self.classify(query=query)

classifier = IntentClassifier()

# --- Tools ---

def get_github_activity(owner: str = "TwingzChenou", repo: str = "CV-Agent") -> str:
    """
    Retrieves recent activity (commits) or README from a public GitHub repository.
    Useful for answering questions about the project's code or recent updates.
    """
    try:
        github_client = GithubClient(github_token=GITHUB_TOKEN)
        loader = GithubRepositoryReader(
            github_client=github_client,
            owner=owner,
            repo=repo,
            use_parser=False,
            verbose=False,
            filter_file_extensions=[".md", ".py", ".js", ".ts", ".jsx", ".tsx"], # Limit to code/docs
            concurrent_requests=2,
        )
        # To be fast, we might strictly want to fetch specific things, but the Reader loads docs.
        # Ideally, for "activity", we'd check commits via API directly, but using the requested Reader:
        # We will load the data. Note: The Reader loads the *content* of the repo.
        # If the user wants "activity" (commits), the Reader might not be the best fit unless we index it.
        # BUT, the instructions say "Charge le repo public... Configure-la pour être rapide".
        # Let's try to limit what we load or use a custom implementation if Reader is too slow.
        # For now, let's load the README and maybe a few top-level files to give context.
        
        # Actually, the user asked for "get_github_activity" but also said "Charge le repo... cible les commits récents ou le README".
        # The GithubRepositoryReader reads files. It doesn't typically read "commits" as a list unless configured/extended.
        # Let's pivot to using the GithubClient directly for commits if the Reader is too heavy, 
        # OR just load the README for context if that's what's meant by "Check activity" in a simple RAG way.
        # Let's stick to a lightweight usage: Load README.
        
        documents = loader.load_data(branch="main") # Loading everything might be slow.
        # Optimization: Filter documents to just README or small set if possible inside load_data? 
        # The Reader supports `ignored_directories` etc.
        
        # Let's manually filter after load if we can't restrict before, or trust the user wants repo content.
        # Getting just the README is safer for speed.
        readme_doc = next((doc for doc in documents if "README.md" in doc.metadata.get("file_path", "")), None)
        
        if readme_doc:
            return f"README Content:\n{readme_doc.text[:5000]}" # Truncate check
        
        return "Could not find README. Loaded " + str(len(documents)) + " files."

    except Exception as e:
        return f"Error fetching GitHub activity: {str(e)}"

def get_cv_info_tool() -> QueryEngineTool:
    """
    Retrieves information from the CV using the Pinecone Vector Store.
    """
    try:
        vector_store = PineconeVectorStore(
            api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX,
        )
        # We assume the index is already populated with dimensionality compatible with the embedding model.
        # Note: We need the Embedding model configured in Settings.
        # Using a standard one or the one used during ingestion. 
        # If ingest used gemini-embedding, we need it here too.
        
        from llama_index.embeddings.gemini import GeminiEmbedding
        Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004")

        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        query_engine = index.as_query_engine()
        
        return QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="cv_query_engine",
                description="Useful for answering questions about Quentin's CV, skills, and experience.",
            ),
        )
    except Exception as e:
        # Return a dummy tool or raise? Agent needs a tool. 
        # Let's return a basic function tool that reports the error as a fallback
        def error_tool(query: str):
            return f"Error connecting to CV database: {str(e)}"
        
        return FunctionTool.from_defaults(fn=error_tool, name="cv_query_engine", description="Error fallback")

# Initialize Agent
github_tool = FunctionTool.from_defaults(fn=get_github_activity, name="get_github_activity")
cv_tool = get_cv_info_tool()

agent = ReActAgent(
    tools=[cv_tool],
    llm=llm,
    verbose=True,
    system_prompt="You are Quentin Forget. You can check his GitHub activity or answer questions about his CV.",
    streaming=False
)

# --- Service Function ---
async def chat_service(query: str):
    """
    Main entry point for handling user queries.
    Uses DSPy to classify intent, then routes to Agent or direct LLM.
    """
    print(f"User Query: {query}")
    
    # 1. Classify Intent
    prediction = classifier(query=query)
    intent = prediction.intent.lower().strip()
    print(f"Predicted Intent: {intent}")

    # 2. Route
    if "chitchat" in intent and "mixed" not in intent:
        # Direct LLM call for speed
        response = llm.complete(query)
        return str(response)
    else:
        # Use Agent for GitHub, CV, or Mixed
        response = await agent.run(query)
        return str(response)

if __name__ == "__main__":
    # Test locally
    import asyncio
    async def main():
        print(await chat_service("Hello!"))
        print(await chat_service("What is in the README of the CV-Agent repo?"))
        # print(await chat_service("What are Quentin's skills?")) # Uncomment if keys are set
    
    asyncio.run(main())
