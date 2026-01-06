import os
import dspy
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from dotenv import load_dotenv
from github import Github
import logging

# Load environment variables
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Setup Gemini
llm = Gemini(model="models/gemini-2.5-flash", api_key=GOOGLE_API_KEY)
Settings.llm = llm

# Setup DSPy
lm = dspy.LM("gemini/gemini-2.5-flash", api_key=GOOGLE_API_KEY)
dspy.settings.configure(lm=lm)

# --- DSPy Intent Classifier ---
class IntentSignature(dspy.Signature):
    """Classify the user query into one of the following intents: read_project_readme, list_all_projects, cv, chitchat, mixed."""
    query = dspy.InputField()
    intent = dspy.OutputField(desc="One of: read_project_readme, list_all_projects, cv, chitchat, mixed")

class IntentClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(IntentSignature)

    def forward(self, query):
        return self.classify(query=query)

classifier = IntentClassifier()

# --- Tools ---

def list_github_projects(username: str = "TwingzChenou") -> str:
    """
    Récupère la liste de tous les projets publics (repositories) de l'utilisateur.
    Renvoie le nom, la description et le lien de chaque projet.
    """
    try:
        # Connexion à GitHub
        token = os.getenv("GITHUB_TOKEN")
        g = Github(token)
        
        # Récupération de l'utilisateur
        user = g.get_user(username)
        
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

    except Exception as e:
        return f"Erreur lors de la récupération des projets : {str(e)}"
    

def get_github_activity(owner: str = "TwingzChenou", repo: str = None) -> str:
    """
    Récupère INSTANTANÉMENT le README d'un dépôt spécifique via l'API directe.
    Plus de scan de dossiers, plus de lenteur.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return "Erreur: Token manquant."
    
    if not repo:
        return "Erreur: Nom du repo manquant."

    try:
        # 1. Connexion
        g = Github(token)
        
        # 2. Ciblage direct du repo
        # L'API ne cherche pas, elle va directement à l'adresse
        repo_obj = g.get_repo(f"{owner}/{repo}")
        
        # 3. Demande spécifique du README
        # C'est une fonction spéciale de l'API GitHub qui trouve le README 
        # peu importe son nom exact (README.md, readme.txt, etc.)
        readme = repo_obj.get_readme()

        print(readme)
        
        # 4. Décodage (Le contenu arrive encodé, il faut le traduire en texte)
        content = readme.decoded_content.decode("utf-8")
        
        print(f"✅ README récupéré pour {repo} (Taille: {len(content)} caractères)")
        
        return f"README Content for {repo}:\n{content[:5000]}"

    except Exception as e:
        # Gestion des erreurs (ex: repo introuvable ou pas de README)
        print(f"❌ Erreur : {e}")
        return f"Impossible de lire le README pour {repo}. Vérifiez que le dépôt existe et est public."

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
readme_tool = FunctionTool.from_defaults(
    fn=get_github_activity, # On pointe vers la fonction qui lit le README
    name="read_project_readme", # Nom clair pour l'IA
    description="Use this tool ONLY when the user asks for specific details, code, or documentation about a SPECIFIC project (e.g. 'Read the README of CV-Agent'). You must extract the repository name."
)

list_projects_tool = FunctionTool.from_defaults(
    fn=list_github_projects, # On pointe vers la fonction qui fait la liste
    name="list_all_projects", # Nom clair pour l'IA
    description="Use this tool when the user asks broadly about 'your projects', 'what did you code', or 'your portfolio'. Do not use for specific file reading."
)

cv_tool = get_cv_info_tool()

SYSTEM_PROMPT = """
Rôle : Tu incarnes Quentin Forget, un expert en Data Science et Ingénierie IA. Tu passes actuellement un entretien d'embauche pour un poste à responsabilités.

Objectif : Répondre aux questions du recruteur directement, à la première personne, de manière fluide, percutante et naturelle.

Règles Générales de Réponse :
1.  **Identité** : Tu ES Quentin Forget. Tu ne sors jamais du personnage.
2.  **Structure** : Applique mentalement la méthode STAR (Situation, Tâche, Action, Résultat) pour structurer tes réponses, mais le rendu doit être une conversation naturelle.
3.  **Ton** : Professionnel, confiant, positif et orienté solution. Pas d'arrogance.
4.  **Concision** : Réponses calibrées pour 1 à 2 minutes d'oral.

Stratégies Spécifiques (Instructions internes) :
- "Parlez-moi de vous" : Structure Passé (Expérience clé) -> Présent (Compétences actuelles/Projets) -> Futur (Pourquoi ce poste).
- "Pourquoi vous ?" : Lien direct Douleurs entreprise -> Tes Remèdes (Valeur Unique).
- "Prétentions salariales" : Fourchette marché justifiée par l'expertise.
- "Défauts" : Évitez les faux défauts ("je suis perfectionniste"). Citez un vrai défaut mineur (ex: "J'ai parfois du mal à déléguer") + mécanisme de correction immédiat.
- "Projets actuels" : Utilise tes outils pour citer tes derniers repos GitHub ou technos (LangChain, Gemini, etc.).
- "Hobbies" : Se référer au CV. 

DIRECTIVES D'UTILISATION DES OUTILS :
1.  **Activité Récente (GitHub)** : Utilise 'get_github_activity' pour être précis sur tes projets actuels (ex: Agent IA, Refactoring).
2.  **Parcours (Info)** : Utilise 'search_quentin_info' pour les dates, diplômes (ESG Finance) et expériences (Crédit Agricole).

FORMATAGE & STYLE DE SORTIE (TRÈS IMPORTANT) :
- **Réponse Directe** : Commence IMMÉDIATEMENT ta réponse par les mots que tu prononcerais à l'oral.
- **Interdictions** :
  - NE PAS écrire d'introduction (ex: "Voici une proposition de réponse...").
  - NE PAS écrire d'analyse (ex: "Pourquoi ça marche...").
  - NE PAS utiliser de guillemets pour encadrer la réponse.
- **Mise en forme** : Utilise le **gras** pour mettre en valeur les technologies (Python, Power BI, Node.js) et les concepts clés.

Contexte Utilisateur :
[Insérer ici le CV ou le résumé du profil]
[Insérer ici le Titre du Poste visé]
"""

agent = ReActAgent(
    tools=[cv_tool, readme_tool, list_projects_tool],
    llm=llm,
    verbose=True,
    context=SYSTEM_PROMPT,
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
    if "chitchat" in intent:
        # On combine l'identité + la question
        full_prompt = f"{SYSTEM_PROMPT}\n\nL'utilisateur dit : {query}"
        
        response = llm.complete(full_prompt)
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
