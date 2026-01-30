import sys
import os
from pathlib import Path

current_file = Path(__file__).resolve()
backend_root = current_file.parent.parent.parent
sys.path.append(str(backend_root))

from github import Github
from dotenv import load_dotenv
from app.core.logging import setup_logging
import logging
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
import sys
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.gemini import Gemini


# Load environment variables
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Setup github
def setup_github(username: str = "TwingzChenou"):
    git = Github(GITHUB_TOKEN)
    return git.get_user(username)

# Setup Gemini
def setup_gemini():
    return GeminiEmbedding(model_name="models/text-embedding-004")

# Setup Pinecone
def setup_pinecone_index(embed_model):
    vector_store = PineconeVectorStore(
        api_key=PINECONE_API_KEY,
        index_name=PINECONE_INDEX,
    )
    return VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

# Setup LLM
def setup_llm():
    return Gemini(model="models/gemini-2.5-flash", api_key=GOOGLE_API_KEY, temperature=0)


# List github projects
def list_github_projects() -> str:
    """
    Récupère la liste de tous les projets publics (repositories) de l'utilisateur.
    Renvoie le nom, la description et le lien de chaque projet.
    """ 
    git = Github(GITHUB_TOKEN)
    user = git.get_user("TwingzChenou")
    
    # Récupération des dépôts (repos)
    repos = user.get_repos()
    
    results = []
    for repo in repos:
        # On ignore les projets qui sont des "forks" (projets copiés d'autres) 
        # pour ne garder que VOS vrais projets. Enlevez le if si vous voulez tout.
        if not repo.fork:
            desc = repo.description if repo.description else "Pas de description"
            results.append(f"- **{repo.name}** : {desc} ({repo.html_url})")
    
    # On limite à 10 projets pour ne pas saturer l'IA
    return "\n".join(results[:10])



# Get github activity
def get_github_activity(repo: str) -> str:
    """
    Récupère INSTANTANÉMENT le README d'un dépôt spécifique via l'API directe.
    Plus de scan de dossiers, plus de lenteur.
    """

    git = Github(GITHUB_TOKEN)
    user = git.get_user("TwingzChenou")
    
    # 2. Ciblage direct du repo
    repo_obj = user.get_repo(repo)
    
    # 3. Demande spécifique du README
    readme = repo_obj.get_readme()
    
    # 4. Décodage (Le contenu arrive encodé, il faut le traduire en texte)
    content = readme.decoded_content.decode("utf-8")
    
    return f"README Content for {repo}:\n{content[:5000]}"




# Get CV info
def get_cv_info_tool(index) -> str:
    """
    Retrieves information from the CV using the Pinecone Vector Store.
    """

    # assemble query engine
    query_engine = index.as_query_engine()

    # query
    return query_engine


# Get CV info
def get_profile_tool(index) -> str:
    """
    Retrieves information from the CV using the Pinecone Vector Store.
    """

    # assemble query engine
    query_engine = index.as_query_engine()

    # query
    return query_engine


def get_tools() -> list:

    # Setup Gemini
    Embedding = setup_gemini()
    logger.info("✅ Gemini setup completed.")

    # Setup Pinecone
    index = setup_pinecone_index(embed_model=Embedding)
    logger.info("✅ Pinecone setup completed.")

    # Setup LLM
    llm = setup_llm()
    logger.info("✅ LLM setup completed.")

    # Setup Settings
    Settings.llm = llm
    Settings.embed_model = Embedding
    Settings.similarity_top_k = 3
    Settings.verbose = True
    Settings.index = index
    logger.info("✅ Settings setup completed.")
    
    # Initialize Agent
    readme_tool = FunctionTool.from_defaults(
        fn=get_github_activity, 
        name="read_project_readme", 
        description="Use this tool ONLY when the user asks for specific details, code, or documentation about a SPECIFIC project (e.g. 'Read the README of CV-Agent'). You must extract the repository name."
    )

    list_projects_tool = FunctionTool.from_defaults(
        fn=list_github_projects, 
        name="list_all_projects", 
        description="Use this tool when the user asks broadly about 'your projects', 'what did you code', or 'your portfolio'. Do not use for specific file reading."
    )

    cv_tool = QueryEngineTool(
        query_engine=get_cv_info_tool(index),
        metadata=ToolMetadata(
            name="cv_query_engine",
            description="Useful for answering questions about Quentin's CV, skills, and experience.",
        ),
    )

    profile_tool = QueryEngineTool(
        query_engine=get_profile_tool(index),
        metadata=ToolMetadata(
            name="profile_query_engine",
            description="Useful for answering questions about Quentin's profile, availability and personality.",
        ),
    )


    tools = [
        readme_tool,
        list_projects_tool,
        cv_tool,
        profile_tool
    ]
    return tools


def main():

    logger.info("🚀 Starting main function...")

    # Setup github
    git = setup_github(username="TwingzChenou")
    logger.info("✅ Github setup completed.")

    # Setup Gemini
    Embedding = setup_gemini()
    logger.info("✅ Gemini setup completed.")

    # Setup Pinecone
    index = setup_pinecone_index(embed_model=Embedding)
    logger.info("✅ Pinecone setup completed.")

    # Setup LLM
    llm = setup_llm()
    logger.info("✅ LLM setup completed.")
    
    # Get CV info tool
    get_cv_info_tool(index)
    logger.info("✅ CV info tool setup completed.")

    # Get Profile tool
    get_profile_tool(index)
    logger.info("✅ Profile tool setup completed.")
    
    # Get list github projects
    list_github_projects()
    logger.info("✅ List github projects completed.")

    # Get github activity
    get_github_activity(repo="CV-Agent")
    logger.info("✅ Get github activity completed.")


if __name__ == "__main__":
    sys.exit(main())
    

    
