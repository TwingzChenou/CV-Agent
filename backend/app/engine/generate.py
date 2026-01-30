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
    intent = dspy.OutputField(desc="One of: read_project_readme(project_name), list_all_projects, cv_query_engine(stacks techniques, diplome, formation, experience professionnelle, compétences, langues, hobbies, localisation), profile_query_engine(salaire, disponibilité, localisation, contrat, personalité, motivation), mixed")

class IntentClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(IntentSignature)

    def forward(self, query):
        return self.classify(query=query)

trainset = [
    # --- Catégorie : profile_query_engine ---
    dspy.Example(query="Quelles sont tes prétentions salariales ?", intent="profile_query_engine(salaire)").with_inputs("query"),
    dspy.Example(query="Es-tu disponible immédiatement ?", intent="profile_query_engine(disponibilité)").with_inputs("query"),
    dspy.Example(query="Quelle est ta localisation ?", intent="profile_query_engine(localisation)").with_inputs("query"),
    dspy.Example(query="Quelles sont vos points forts et vos points faibles ?", intent="profile_query_engine(personalité)").with_inputs("query"),
    dspy.Example(query="Quelle est ta motivation ?", intent="profile_query_engine(motivation)").with_inputs("query"),
    dspy.Example(query="Où vous voyez-vous dans 5 ans ?", intent="profile_query_engine(motivation)").with_inputs("query"),
    dspy.Example(query="Quel est votre contrat ?", intent="profile_query_engine(contrat)").with_inputs("query"),
    
    
    # --- Catégorie : direct_answer (Infos du System Prompt) ---

    # --- Catégorie : chitchat (Infos du System Prompt) ---
    dspy.Example(query="Salut, comment ça va ?", intent="chitchat").with_inputs("query"),
    

    # --- Catégorie : cv (Infos complexes nécessitant recherche) ---
    dspy.Example(query="Détaille-moi ton expérience chez Crédit Agricole", intent="cv (experience)").with_inputs("query"),
    dspy.Example(query="Quelles sont tes stack techniques ?", intent="cv (stacks techniques)").with_inputs("query"),
    dspy.Example(query="Quelle est ta formation ?", intent="cv (formation)").with_inputs("query"),
    dspy.Example(query="Quelle est ta diplome ?", intent="cv (diplome)").with_inputs("query"),
    dspy.Example(query="Quelle est ta localisation ?", intent="cv (localisation)").with_inputs("query"),
    dspy.Example(query="Quels sont tes hobbies ?", intent="cv (hobbies)").with_inputs("query"),
    
    # --- Autres catégories ---
    dspy.Example(query="Montre moi tes projets github", intent="list_all_projects").with_inputs("query"),
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
        print("--- TEST 1 : Chitchat ---")
        print(await generate_response("Bonjour, comment ça va ?"))
        
        print("\n--- TEST 2 : CV (RAG) ---")
        print(await generate_response("Quelles stacks techniques maitrise-t-il ?"))
        
        print("\n--- TEST 3 : GitHub ---")
        print(await generate_response("Quels sont ses projets GitHub ?"))

        print("\n--- TEST 4 : Profile (RAG) ---")
        print(await generate_response("Quelles sont ses disponibilités ?"))

        print("\n--- TEST 5 : Unformal question ---")
        print(await generate_response("Tu fais quoi dans la vie ?"))

        print("\n--- TEST 6 : Question for J.A.R.V.I.S ---")
        print(await generate_response("Tu fais quoi dans la vie J.A.R.V.I.S?"))
    
    asyncio.run(main())

