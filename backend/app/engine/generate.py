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
from app.engine.tools import get_tools, setup_pinecone_index, setup_gemini, setup_llm, get_github_client
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

llm = setup_llm()

# Setup DSPy
lm = dspy.LM("gemini/gemini-2.5-flash", api_key=GOOGLE_API_KEY, temperature=0)
dspy.settings.configure(lm=lm)

# --- DSPy Intent Classifier ---
class IntentSignature(dspy.Signature):
    """Classify the user query into one of the following intents:
    - read_project_readme(project_name) : Read the README of a specific GitHub project.
    - list_all_projects : List all projects or portfolio.
    - cv_query_engine(search_query) : Search the CV for specific information.
      CRITICAL rules for search_query:
      * For ANY query asking for contacts, contact information, email, phone number, address, or how to contact Quentin, search_query MUST be exactly "contacts". Do NOT use "coordonnées de contact" or "coordonnées".
      * For general skills, areas of expertise, or overall competences, search_query MUST be exactly "Competences".
      * For studies, education, university, degrees, or formations, search_query MUST be exactly "études".
      * For specific technical skills or specific technologies (e.g. 'Deep Learning', 'NLP', 'Java', 'Machine Learning'), extract the exact technology name.
    - chitchat : Small talk, greetings, or questions about J.A.R.V.I.S himself.
    - mixed : Multiple intents or complex queries.
    """
    query = dspy.InputField()
    intent = dspy.OutputField(desc="The classified intent, using the exact format: read_project_readme(project_name), list_all_projects, cv_query_engine(search_query), chitchat, or mixed. Standardize search_query according to the rules.")

class IntentClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(IntentSignature)

    def forward(self, query):
        return self.classify(query=query)

trainset = [
    # --- Catégorie : cv (Extraction de mots-clés/sujets pour le RAG) ---
    dspy.Example(query="Quelles sont ses prétentions salariales ?", intent="cv_query_engine(salaire)").with_inputs("query"),
    dspy.Example(query="Quelles sont ses disponibilités ?", intent="cv_query_engine(disponibilités)").with_inputs("query"),
    dspy.Example(query="Quels sont ses points forts et ses points faibles ?", intent="cv_query_engine(personalité)").with_inputs("query"),
    dspy.Example(query="Quelle est sa motivation ?", intent="cv_query_engine(motivation)").with_inputs("query"),
    dspy.Example(query="Où se voit-il dans 5 ans ?", intent="cv_query_engine(motivation)").with_inputs("query"),
    dspy.Example(query="Quel est son contrat ?", intent="cv_query_engine(contrat)").with_inputs("query"),
    dspy.Example(query="Détaille-moi son expérience chez Crédit Agricole", intent="cv_query_engine(experience)").with_inputs("query"),
    dspy.Example(query="Quelles sont ses stack techniques ?", intent="cv_query_engine(stacks techniques)").with_inputs("query"),
    dspy.Example(query="Quelle est sa formation ?", intent="cv_query_engine(études)").with_inputs("query"),
    dspy.Example(query="Quelles sont ses diplomes ?", intent="cv_query_engine(études)").with_inputs("query"),
    dspy.Example(query="Quels sont ses hobbies ?", intent="cv_query_engine(hobbies)").with_inputs("query"),
    dspy.Example(query="Quelle est sa localisation ?", intent="cv_query_engine(localisation)").with_inputs("query"),
    
    # Exemples dynamiques pour prouver la généralisation du paramètre de recherche
    dspy.Example(query="Quentin Forget a-t-il d'expérience en Deep Learning ?", intent="cv_query_engine(Deep Learning)").with_inputs("query"),
    dspy.Example(query="Quels sont les compétences de Quentin en NLP ?", intent="cv_query_engine(NLP)").with_inputs("query"),
    dspy.Example(query="Quels sont les domaines de compétence de Quentin Forget ?", intent="cv_query_engine(Competences)").with_inputs("query"),
    dspy.Example(query="Quels sont les coordonnées de contact de Quentin Forget ?", intent="cv_query_engine(contacts)").with_inputs("query"),
    dspy.Example(query="Affiche mes coordonnées de contact", intent="cv_query_engine(contacts)").with_inputs("query"),
    dspy.Example(query="Quentin a-t-il des coordonnées de contact ?", intent="cv_query_engine(contacts)").with_inputs("query"),
    dspy.Example(query="Pourrez-vous me donner les coordonnées de contact de Quentin?", intent="cv_query_engine(contacts)").with_inputs("query"),
    dspy.Example(query="Comment puis-je contacter Quentin Forget ?", intent="cv_query_engine(contacts)").with_inputs("query"),
    dspy.Example(query="Quelles sont les langues parlées par Quentin Forget ?", intent="cv_query_engine(langues)").with_inputs("query"),
    dspy.Example(query="Est-ce que Quentin a déjà des projets sur GitHub en Java ?", intent="cv_query_engine(Java)").with_inputs("query"),
    dspy.Example(query="Quelles sont les compétences de Quentin en Machine Learning ?", intent="cv_query_engine(Machine Learning)").with_inputs("query"),

    # --- Catégorie : chitchat (Infos du System Prompt) ---
    dspy.Example(query="Salut, comment ça va ?", intent="chitchat").with_inputs("query"),

    # --- Autres catégories ---
    dspy.Example(query="Montre moi ses projets github", intent="list_all_projects").with_inputs("query"),
    dspy.Example(query="Lis le README du projet CV-Agent", intent="read_project_readme(CV-Agent)").with_inputs("query"),
]

# Compilation du modèle
print("🧠 Optimisation du classifieur d'intentions DSPy...")
teleprompter = LabeledFewShot(k=5) # k = nombre d'exemples à utiliser dans le prompt
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

class MockToolCall:
    def __init__(self, tool_name, tool_kwargs):
        self.tool_name = tool_name
        self.tool_kwargs = tool_kwargs

class JarvisAgentResponse:
    def __init__(self, response_text, tool_calls=None, source_nodes=None):
        self.response = response_text
        self.tool_calls = tool_calls or []
        self.source_nodes = source_nodes or []
        
    def __str__(self):
        return str(self.response)

def parse_intent(intent_str):
    import re
    match = re.match(r"^([\w_]+)(?:\((.*)\))?$", intent_str)
    if match:
        name = match.group(1)
        arg = match.group(2)
        if arg is not None:
            arg = arg.strip()
            if not arg:
                arg = None
        return name, arg
    return intent_str, None

class JarvisAgent:
    def __init__(self, react_agent, classifier, llm, pinecone_index):
        self.react_agent = react_agent
        self.classifier = classifier
        self.llm = llm
        self.pinecone_index = pinecone_index

    async def run(self, agent_input, *args, **kwargs):
        # Extract the user's raw query from the control prompt sandwich if present
        query = agent_input.split("### DIRECTIVE DE CONTRÔLE ###")[0].strip()
        
        # 1. Classification
        try:
            intent_output = self.classifier(query)
            intent_str = str(intent_output.intent).strip()
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            intent_str = "mixed"
            
        logger.info(f"JarvisAgent Routing | Query: {query} | Intent: {intent_str}")
        name, arg = parse_intent(intent_str)
        
        # 2. Normalize routing names and arguments generally (not hardcoded to test dataset)
        if name:
            name = name.strip()
        if arg:
            arg = arg.strip()
            
        # Case-insensitive resolution of repository name to make it generic
        if name == "read_project_readme" and arg:
            try:
                git = get_github_client()
                user = git.get_user("TwingzChenou")
                repos = user.get_repos()
                for r in repos:
                    if r.name.lower() == arg.lower():
                        arg = r.name
                        break
            except Exception as e:
                logger.error(f"Error resolving repository name case-insensitively: {e}")
                

        # 3. Direct Routing
        if name == "chitchat":
            chitchat_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"L'utilisateur dit : \"{query}\"\n"
                f"En tant que J.A.R.V.I.S., réponds de façon courtoise, flegmatique et appropriée (en parlant de Monsieur Forget à la 3ème personne si besoin, et à la 1ère personne pour toi-même) :"
            )
            chitchat_response = await self.llm.acomplete(chitchat_prompt)
            return JarvisAgentResponse(response_text=str(chitchat_response))
            
        elif name == "list_all_projects":
            from app.engine.tools import list_github_projects
            try:
                raw_projects = list_github_projects()
            except Exception as e:
                logger.error(f"Error listing projects: {e}")
                raw_projects = "Aucun projet public trouvé ou indisponible pour le moment."
                
            rephrase_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Voici les données brutes sur les projets de Monsieur Forget :\n"
                f"{raw_projects}\n\n"
                f"Rédige une réponse parfaite au ton de J.A.R.V.I.S. pour présenter ces projets à la requête : \"{query}\" :"
            )
            rephrased_response = await self.llm.acomplete(rephrase_prompt)
            
            tool_calls = [MockToolCall("list_all_projects", {})]
            return JarvisAgentResponse(response_text=str(rephrased_response), tool_calls=tool_calls)
            
        elif name == "read_project_readme" and arg:
            from app.engine.tools import get_github_activity
            try:
                raw_readme = get_github_activity(arg)
            except Exception as e:
                logger.error(f"Error reading README for {arg}: {e}")
                raw_readme = f"README ou dépôt '{arg}' introuvable."
                
            rephrase_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Voici le contenu du README pour le projet {arg} :\n"
                f"{raw_readme}\n\n"
                f"Rédige une réponse parfaite au ton de J.A.R.V.I.S. pour résumer ou expliquer ce projet en réponse à la requête : \"{query}\" :"
            )
            rephrased_response = await self.llm.acomplete(rephrase_prompt)
            
            tool_calls = [MockToolCall("read_project_readme", {"repo": arg})]
            return JarvisAgentResponse(response_text=str(rephrased_response), tool_calls=tool_calls)
            
        elif name == "cv_query_engine" and arg:
            # Query Pinecone directly using the query engine
            query_engine = self.pinecone_index.as_query_engine()
            try:
                retrieved_response = await query_engine.aquery(arg)
                response_content = str(retrieved_response)
                source_nodes = retrieved_response.source_nodes if hasattr(retrieved_response, "source_nodes") else []
            except Exception as e:
                logger.error(f"Error querying RAG: {e}")
                response_content = "Information non trouvée."
                source_nodes = []
                
            rephrase_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Voici l'information trouvée dans le CV de Monsieur Forget :\n"
                f"{response_content}\n\n"
                f"Rédige une réponse parfaite au ton de J.A.R.V.I.S. pour répondre à la question de l'utilisateur : \"{query}\" :"
            )
            rephrased_response = await self.llm.acomplete(rephrase_prompt)
            
            tool_calls = [MockToolCall("cv_query_engine", {"input": arg})]
            return JarvisAgentResponse(
                response_text=str(rephrased_response), 
                tool_calls=tool_calls, 
                source_nodes=source_nodes
            )
            
        # 3. Fallback to ReActAgent
        logger.info(f"Fallback to ReActAgent for query: {query}")
        return await self.react_agent.run(agent_input)

# Setup wrapper objects
react_agent = ReActAgent(
    tools=get_tools(),
    llm=llm,
    verbose=True,
    context=SYSTEM_PROMPT,
    streaming=False
)

embed_model = setup_gemini()
pinecone_index = setup_pinecone_index(embed_model)

agent = JarvisAgent(
    react_agent=react_agent,
    classifier=classifier,
    llm=llm,
    pinecone_index=pinecone_index
)

async def generate_response(query):
    logger.info(f"Query: {query}")
    
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
    
        print("\n--- TEST 3 : Github ---")
        print(await generate_response("Quels sont les projets de Github de Quentin ?"))
        
        print("\n--- TEST 4 : GitHub ---")
        print(await generate_response("Décris moi son projet CV-Agent ?"))

        print("\n--- TEST 5 : Profile (RAG) ---")
        print(await generate_response("Quels sont les contacts de Quentin ?"))

        print("\n--- TEST 6 : Unformal question ---")
        print(await generate_response("Tu fais quoi dans la vie ?"))

        print("\n--- TEST 7 : Question for J.A.R.V.I.S ---")
        print(await generate_response("Tu fais quoi dans la vie J.A.R.V.I.S?"))
    
    asyncio.run(main())

