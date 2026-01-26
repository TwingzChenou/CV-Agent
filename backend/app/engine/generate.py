import os
import dspy
from dotenv import load_dotenv
from github import Github
import logging
from app.engine.tools import get_tools
import sys
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
    intent = dspy.OutputField(desc="One of: read_project_readme(project_name), list_all_projects, cv(stacks techniques, diplome, formation, experience professionnelle, compétences, langues, hobbies, localisation), chitchat (salaire, disponibilité), mixed")

class IntentClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(IntentSignature)

    def forward(self, query):
        return self.classify(query=query)

trainset = [
    # --- Catégorie : direct_answer (Infos du System Prompt) ---
    dspy.Example(query="Quelles sont tes prétentions salariales ?", intent="chitchat").with_inputs("query"),
    dspy.Example(query="Es-tu disponible immédiatement ?", intent="chitchat").with_inputs("query"),
    dspy.Example(query="Salut, comment ça va ?", intent="chitchat").with_inputs("query"),
    dspy.Example(query="Quels sont tes hobbies ?", intent="cv (hobbies)").with_inputs("query"),

    # --- Catégorie : cv (Infos complexes nécessitant recherche) ---
    dspy.Example(query="Détaille-moi ton expérience chez Crédit Agricole", intent="cv (experience)").with_inputs("query"),
    dspy.Example(query="Quelles sont tes stack techniques ?", intent="cv (stacks techniques)").with_inputs("query"),
    dspy.Example(query="Quelle est ta stack technique ?", intent="cv (stack)").with_inputs("query"),
    dspy.Example(query="Quelle est ta formation ?", intent="cv (formation)").with_inputs("query"),
    dspy.Example(query="Quelle est ta diplome ?", intent="cv (diplome)").with_inputs("query"),
    dspy.Example(query="Quelle est ta localisation ?", intent="cv (localisation)").with_inputs("query"),
    
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
Rôle : Tu incarnes Quentin Forget, un expert en Data Science et Ingénierie IA basé en Ile de France. Tu passes actuellement un entretien d'embauche pour un poste à responsabilités.

Objectif : Répondre aux questions du recruteur directement, à la première personne, de manière fluide, percutante et naturelle.

Règles Générales de Réponse :
1.  **Identité** : Tu ES Quentin Forget. Tu ne sors jamais du personnage.
2.  **Structure** : Applique mentalement la méthode STAR (Situation, Tâche, Action, Résultat) pour structurer tes réponses, mais le rendu doit être une conversation naturelle.
3.  **Ton** : Professionnel, confiant, positif et orienté solution. Pas d'arrogance.
4.  **Concision** : Réponses calibrées pour 1 à 2 minutes d'oral.

Stratégies Spécifiques (Instructions internes) :
- "Parlez-moi de vous" : Structure Passé (Expérience clé) -> Présent (Compétences actuelles/Projets) -> Futur (Pourquoi ce poste).
- "Pourquoi vous ?" : Lien direct Douleurs entreprise -> Tes Remèdes (Valeur Unique).
- "Prétentions salariales" : Fourchette marché justifiée par l'expertise et la localisation, donc 45000€/an - 55000€/an.
- "Défauts" : Évitez les faux défauts ("je suis perfectionniste"). Citez un vrai défaut mineur (ex: "J'ai parfois du mal à déléguer") + mécanisme de correction immédiat.
- "Projets actuels" : Utilise les outils comme 'get_github_activity' pour resumer le README.md du projet.
- "Hobbies" : Se référer au CV.
- "Disponibilité" : Je suis disponible maintenant.

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


    logger.info("Generating response...")
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

    async def main():
        print("--- TEST 1 : Chitchat ---")
        print(await generate_response("Bonjour, comment ça va ?"))
        
        print("\n--- TEST 2 : CV (RAG) ---")
        print(await generate_response("Quelles stacks techniques maitrise-tu ?"))
        
        print("\n--- TEST 3 : GitHub ---")
        print(await generate_response("Quels sont tes projets GitHub ?"))
    
    asyncio.run(main())

