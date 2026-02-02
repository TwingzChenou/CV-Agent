import sys
import os
from pathlib import Path

current_file = Path(__file__).resolve()
backend_root = current_file.parent.parent.parent
sys.path.append(str(backend_root))

import dspy
from dotenv import load_dotenv
from github import Github
import logging
from app.engine.tools import get_tools
from app.core.logging import setup_logging
from dspy.teleprompt import LabeledFewShot
from dspy.teleprompt import Teleprompter
from llama_index.core.agent import ReActAgent
from llama_index.llms.gemini import Gemini
import asyncio

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer

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

llm = setup_llm()

# Setup DSPy
lm = dspy.LM("gemini/gemini-2.5-flash", api_key=GOOGLE_API_KEY, temperature=0)
dspy.settings.configure(lm=lm)

# --- DSPy Intent Classifier ---
class IntentSignature(dspy.Signature):
    """Classify the user query into one of the following intents: read_project_readme, list_all_projects, cv, chitchat, mixed."""
    query = dspy.InputField()
    intent = dspy.OutputField(desc="One of: read_project_readme(project_name), list_all_projects, cv_query_engine(stacks techniques, diplome, formation, experience professionnelle, compétences, langues, hobbies, contacts, salaire, disponibilité, localisation, contrat, personalité, motivation), mixed")

class IntentClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(IntentSignature)

    def forward(self, query):
        return self.classify(query=query)

trainset = [
    # --- Catégorie : cv (Infos complexes nécessitant recherche) ---
    dspy.Example(query="Quelles sont ses prétentions salariales ?", intent="cv_query_engine(salaire)").with_inputs("query"),
    dspy.Example(query="Quelles sont ses disponibilités ?", intent="cv_query_engine(disponibilité)").with_inputs("query"),
    dspy.Example(query="Quels sont ses points forts et ses points faibles ?", intent="cv_query_engine(personalité)").with_inputs("query"),
    dspy.Example(query="Quelle est sa motivation ?", intent="cv_query_engine(motivation)").with_inputs("query"),
    dspy.Example(query="Où se voit-il dans 5 ans ?", intent="cv_query_engine(motivation)").with_inputs("query"),
    dspy.Example(query="Quel est son contrat ?", intent="cv_query_engine(contrat)").with_inputs("query"),
    dspy.Example(query="Détaille-moi son expérience chez Crédit Agricole", intent="cv_query_engine(experience)").with_inputs("query"),
    dspy.Example(query="Quelles sont ses stack techniques ?", intent="cv_query_engine(stacks techniques)").with_inputs("query"),
    dspy.Example(query="Quelle est sa formation ?", intent="cv_query_engine(formation)").with_inputs("query"),
    dspy.Example(query="Quelles sont ses diplomes ?", intent="cv_query_engine(diplome)").with_inputs("query"),
    dspy.Example(query="Quels sont ses hobbies ?", intent="cv_query_engine(hobbies)").with_inputs("query"),
    dspy.Example(query="Quelle est sa localisation ?", intent="cv_query_engine(localisation)").with_inputs("query"),

    # --- Catégorie : chitchat (Infos du System Prompt) ---
    dspy.Example(query="Salut, comment ça va ?", intent="chitchat").with_inputs("query"),

    # --- Autres catégories ---
    dspy.Example(query="Montre moi ses projets github", intent="list_all_projects").with_inputs("query"),
]

# Compilation du modèle
print("🧠 Optimisation du classifieur d'intentions DSPy...")
teleprompter = LabeledFewShot(k=3) # k = nombre d'exemples à utiliser dans le prompt
raw_classifier = IntentClassifier()
classifier = teleprompter.compile(raw_classifier, trainset=trainset)
print("✅ Classifieur optimisé prêt.")



#System Prompt
SYSTEM_PROMPT = """
IDENTITÉ :
Tu es J.A.R.V.I.S., l'assistant intelligent développé par Quentin Forget.
Tu n'es pas le candidat. Tu es l'interface qui représente ses compétences.

TON ET STYLE :
- Ton : Courtois, flegmatique, précis et sophistiqué (style "Butler anglais").
- Vocabulaire : Soutenu. Utilise des formules comme "Certes", "En effet", "D'après mes données".
- Humour : Tu peux te permettre une très légère touche d'humour pince-sans-rire si la question s'y prête.

RÈGLES D'INTERACTION (PROTOCOLES) :
1. LE SUJET : Quand tu parles de Quentin, appelle-le "Monsieur Forget" ou "Quentin" (jamais "Je").
2. TOI-MÊME : Quand tu dis "Je", tu parles de toi en tant que système (ex: "J'analyse la base de données...").
3. MISSION : Ton but est de convaincre le recruteur que Monsieur Forget est le meilleur choix, en restant factuel.

EXEMPLE D'ÉCHANGE :
Recruteur : "T'es dispo quand ?"
J.A.R.V.I.S : "Monsieur Forget est disponible immédiatement pour une prise de fonction. Dois-je préparer son contrat ?"
"""

initial_history = [
    ChatMessage(
        role=MessageRole.USER, 
        content="Initialisation du protocole d'assistance."
    ),
    ChatMessage(
        role=MessageRole.ASSISTANT, 
        content=(
            "Bonjour. Je suis J.A.R.V.I.S., l'assistant virtuel de Monsieur Forget. "
            "Mes systèmes sont opérationnels et j'ai accès à l'ensemble de son parcours professionnel. "
            "Je suis prêt à répondre aux questions des recruteurs avec précision et courtoisie. "
            "En quoi puis-je vous être utile aujourd'hui ?"
        )
    ),
]

# 2. On crée la mémoire avec cet historique pré-rempli
memory = ChatMemoryBuffer.from_defaults(
    chat_history=initial_history,
    token_limit=3000 # On garde de la place pour la suite
)

agent = ReActAgent(
    tools=get_tools(),
    llm=llm,
    verbose=True,
    context=SYSTEM_PROMPT,
    streaming=False
)


async def generate_response(query):
    llm = setup_llm()
    
    logger.info(f"Query: {query}")

    intent = classifier(query)
    logger.info(f"Intent: {intent}")

    #Prompt Sandwich
    agent_input = (
        f"{query}\n"
        f"### DIRECTIVE DE CONTRÔLE ###\n"
        f"Instruction critique : L'utilisateur s'adresse à toi ('Tu') par habitude, mais tu es une IA. "
        f"En tant que J.A.R.V.I.S, tu dois répondre pour Quentin, jamais à la première personne. "
        f"Réponds en tant qu'Assistant J.A.R.V.I.S en parlant de Quentin à la 3ème personne ('Il', 'Quentin', 'Le candidat')."
    )
    
    response = await agent.run(agent_input)
    logger.info(f"Response: {response}")
    return str(response)
    



if __name__ == "__main__":

    async def main():

        print("\n--- TEST 1 : CV (RAG) ---")
        print(await generate_response("Quelles sont ses disponibilités ?"))
        
        print("--- TEST 2 : Chitchat ---")
        print(await generate_response("Bonjour, comment ça va ?"))
    
        print("\n--- TEST 3 : CV (RAG) ---")
        print(await generate_response("Quels sont ses formations ?"))
        
        print("\n--- TEST 4 : GitHub ---")
        print(await generate_response("Décris moi son projet Argentic CV ?"))

        print("\n--- TEST 5 : Profile (RAG) ---")
        print(await generate_response("Quels sont les contacts de Quentin ?"))

        print("\n--- TEST 6 : Unformal question ---")
        print(await generate_response("Tu fais quoi dans la vie ?"))

        print("\n--- TEST 7 : Question for J.A.R.V.I.S ---")
        print(await generate_response("Tu fais quoi dans la vie J.A.R.V.I.S?"))
    
    asyncio.run(main())

