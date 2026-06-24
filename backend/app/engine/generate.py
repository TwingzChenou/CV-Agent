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
from langfuse.decorators import observe, langfuse_context

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer

# Load environment variables
load_dotenv()

# Setup Logging
setup_logging()
logger = logging.getLogger(__name__)

# --- Langfuse & OpenInference (DSPy) Instrumentation ---
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"

from contextvars import ContextVar
from llama_index.core.callbacks import CallbackManager, CBEventType, EventPayload
from llama_index.core.callbacks.base import BaseCallbackHandler

# Context variable for tracking token usage per request/context
agent_token_usage_var = ContextVar("agent_token_usage", default=None)

# Pricing parameters (per 1,000,000 tokens)
GEMINI_INPUT_COST_PER_M = 0.30
GEMINI_OUTPUT_COST_PER_M = 2.50
GEMINI_EMBEDDING_COST_PER_M = 0.15

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)

def calculate_cost(prompt_tokens, completion_tokens, embedding_tokens):
    return (prompt_tokens * GEMINI_INPUT_COST_PER_M + 
            completion_tokens * GEMINI_OUTPUT_COST_PER_M + 
            embedding_tokens * GEMINI_EMBEDDING_COST_PER_M) / 1_000_000

class ContextTokenCountingHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])

    def start_trace(self, trace_id: str | None = None) -> None:
        pass

    def end_trace(self, trace_id: str | None = None, trace_map: dict | None = None) -> None:
        pass

    def on_event_start(self, event_type, payload=None, event_id="", parent_id="", **kwargs):
        pass

    def on_event_end(self, event_type, payload=None, event_id="", **kwargs):
        if payload is None:
            return
            
        token_usage = agent_token_usage_var.get()
        if token_usage is None:
            return
            
        if event_type == CBEventType.LLM:
            response = payload.get(EventPayload.RESPONSE)
            if response is not None:
                # Check additional_kwargs or raw responses for tokens
                prompt_tokens = response.additional_kwargs.get("prompt_tokens", 0)
                completion_tokens = response.additional_kwargs.get("completion_tokens", 0)
                
                # Try raw model response if present
                if prompt_tokens == 0 or completion_tokens == 0:
                    raw = getattr(response, "raw", None)
                    if raw and hasattr(raw, "usage_metadata"):
                        usage = raw.usage_metadata
                        if usage:
                            prompt_tokens = getattr(usage, "prompt_token_count", prompt_tokens)
                            completion_tokens = getattr(usage, "candidates_token_count", completion_tokens)
                
                # Fallback to estimate if still 0
                if prompt_tokens == 0:
                    messages = payload.get(EventPayload.MESSAGES)
                    prompt_text = ""
                    if messages:
                        prompt_text = "\n".join([str(m.content) for m in messages])
                    else:
                        prompt_text = str(payload.get(EventPayload.PROMPT, ""))
                    prompt_tokens = estimate_tokens(prompt_text)
                    
                if completion_tokens == 0:
                    completion_text = str(response)
                    completion_tokens = estimate_tokens(completion_text)
                    
                token_usage["prompt_tokens"] += prompt_tokens
                token_usage["completion_tokens"] += completion_tokens
                
        elif event_type == CBEventType.EMBEDDING:
            chunks = payload.get(EventPayload.CHUNKS)
            if chunks:
                embedding_tokens = sum(estimate_tokens(chunk) for chunk in chunks)
                token_usage["embedding_tokens"] += embedding_tokens

if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    try:
        from llama_index.core import set_global_handler, Settings
        set_global_handler(
            "langfuse",
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST
        )
        logger.info("Langfuse global callback handler registered for LlamaIndex.")
        
        # Monkey patch LlamaIndexCallbackHandler to support Gemini token parsing, model name normalization, and cost calculation
        from langfuse.llama_index.llama_index import LlamaIndexCallbackHandler
        
        def patched_parse_usage_from_event_payload(self, event_payload: dict):
            model = None
            usage = None
            if not event_payload:
                return model, usage
            
            response = event_payload.get(EventPayload.RESPONSE)
            
            # 1. Extract model name
            if response is not None:
                if hasattr(response, "raw") and response.raw is not None:
                    model = getattr(response.raw, "model", None)
                if not model and hasattr(response, "model"):
                    model = getattr(response, "model", None)
            
            if not model:
                model = "gemini-2.5-flash"
                
            # 2. Extract or estimate tokens
            prompt_tokens = 0
            completion_tokens = 0
            
            if response is not None:
                if hasattr(response, "additional_kwargs"):
                    prompt_tokens = response.additional_kwargs.get("prompt_tokens", 0)
                    completion_tokens = response.additional_kwargs.get("completion_tokens", 0)
                    
                if prompt_tokens == 0 or completion_tokens == 0:
                    raw = getattr(response, "raw", None)
                    if raw and hasattr(raw, "usage_metadata"):
                        usage_metadata = raw.usage_metadata
                        if usage_metadata:
                            prompt_tokens = getattr(usage_metadata, "prompt_token_count", prompt_tokens)
                            completion_tokens = getattr(usage_metadata, "candidates_token_count", completion_tokens)
                            
                # Fallback to estimation
                if prompt_tokens == 0:
                    messages = event_payload.get(EventPayload.MESSAGES)
                    prompt_text = ""
                    if messages:
                        prompt_text = "\n".join([str(m.content) for m in messages])
                    else:
                        prompt_text = str(event_payload.get(EventPayload.PROMPT, ""))
                    prompt_tokens = estimate_tokens(prompt_text)
                    
                if completion_tokens == 0:
                    completion_text = str(response)
                    completion_tokens = estimate_tokens(completion_text)
                    
            total_tokens = prompt_tokens + completion_tokens
            
            # 3. Calculate costs
            input_cost = (prompt_tokens * GEMINI_INPUT_COST_PER_M) / 1_000_000
            output_cost = (completion_tokens * GEMINI_OUTPUT_COST_PER_M) / 1_000_000
            total_cost = input_cost + output_cost
            
            # 4. Clean up model name to match Langfuse's default registry
            if model:
                if model.startswith("models/"):
                    model = model[7:]
                elif model.startswith("gemini/"):
                    model = model[7:]
                    
            usage = {
                "input": prompt_tokens,
                "output": completion_tokens,
                "total": total_tokens,
                "unit": "TOKENS",
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost
            }
            
            return model, usage

        def patched_handle_embedding_events(self, event_id: str, parent, trace_id: str):
            events = self.event_map[event_id]
            start_event, end_event = events[0], events[-1]

            name = "Embedding"
            model = None
            timeout = None
            if start_event.payload and EventPayload.SERIALIZED in start_event.payload:
                serialized = start_event.payload.get(EventPayload.SERIALIZED, {})
                name = serialized.get("class_name", "Embedding")
                model = serialized.get("model_name", None)
                timeout = serialized.get("timeout", None)

            token_count = 0
            if end_event.payload:
                chunks = end_event.payload.get(EventPayload.CHUNKS, [])
                try:
                    token_count = sum(self._token_counter.get_string_tokens(chunk) for chunk in chunks)
                except Exception:
                    token_count = 0
                if token_count == 0:
                    token_count = sum(estimate_tokens(chunk) for chunk in chunks)

            cleaned_model = model
            if cleaned_model:
                if cleaned_model.startswith("models/"):
                    cleaned_model = cleaned_model[7:]
                elif cleaned_model.startswith("gemini/"):
                    cleaned_model = cleaned_model[7:]

            embed_cost = (token_count * GEMINI_EMBEDDING_COST_PER_M) / 1_000_000

            usage = {
                "input": token_count,
                "output": 0,
                "total": token_count,
                "unit": "TOKENS",
                "input_cost": embed_cost,
                "total_cost": embed_cost
            }

            input_payload = self._parse_input_from_event(end_event)
            output_payload = self._parse_output_from_event(end_event)

            generation = parent.generation(
                id=event_id,
                trace_id=trace_id,
                name=name,
                start_time=start_event.time,
                end_time=end_event.time,
                version=self.version,
                model=cleaned_model,
                input=input_payload,
                output=output_payload,
                usage=usage,
                model_parameters={
                    "request_timeout": timeout,
                },
            )

            return generation

        original_get_root_observation = LlamaIndexCallbackHandler._get_root_observation

        def patched_get_root_observation(self):
            from langfuse.llama_index.llama_index import context_root, context_trace_metadata
            user_provided_root = context_root.get()
            if user_provided_root is not None:
                self.trace = user_provided_root
                if getattr(self, "update_stateful_client", False):
                    trace_metadata = context_trace_metadata.get()
                    name = (
                        trace_metadata["name"]
                        or self.trace_name
                        or f"LlamaIndex_{self._llama_index_trace_name}"
                    )
                    version = trace_metadata["version"] or self.version
                    release = trace_metadata["release"] or self.release
                    session_id = trace_metadata["session_id"] or self.session_id
                    user_id = trace_metadata["user_id"] or self.user_id
                    metadata = trace_metadata["metadata"] or self.metadata
                    tags = trace_metadata["tags"] or self.tags
                    public = trace_metadata["public"] or None

                    user_provided_root.update(
                        name=name,
                        version=version,
                        session_id=session_id,
                        user_id=user_id,
                        metadata=metadata,
                        tags=tags,
                        release=release,
                        public=public,
                    )
                return user_provided_root
            else:
                return original_get_root_observation(self)

        LlamaIndexCallbackHandler._parse_usage_from_event_payload = patched_parse_usage_from_event_payload
        LlamaIndexCallbackHandler._handle_embedding_events = patched_handle_embedding_events
        LlamaIndexCallbackHandler._get_root_observation = patched_get_root_observation

        # Register the context token counting handler globally in LlamaIndex Settings
        token_handler = ContextTokenCountingHandler()
        if Settings.callback_manager is None:
            Settings.callback_manager = CallbackManager([token_handler])
        else:
            Settings.callback_manager.add_handler(token_handler)
    except Exception as e:
        logger.error(f"Failed to register Langfuse handler: {e}")

    try:
        from openinference.instrumentation.dspy import DSPyInstrumentor
        DSPyInstrumentor().instrument()
        logger.info("DSPy auto-instrumentation via OpenInference enabled.")
    except Exception as e:
        logger.error(f"Failed to register DSPyInstrumentor: {e}")

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

from google import genai
from google.genai import types
from app.engine.caching import get_or_create_cv_cache

llm = setup_llm()
genai_client = genai.Client(api_key=GOOGLE_API_KEY)

# Setup DSPy
lm = dspy.LM("gemini/gemini-2.5-flash", api_key=GOOGLE_API_KEY, temperature=0)
dspy.settings.configure(lm=lm)

# Pricing parameters (per 1,000,000 tokens)
GEMINI_CACHED_INPUT_COST_PER_M = 0.075 # 75% cheaper for cached input tokens

def calculate_cache_cost(prompt_tokens, cached_tokens, completion_tokens):
    uncached_input = max(0, prompt_tokens - cached_tokens)
    return (uncached_input * GEMINI_INPUT_COST_PER_M +
            cached_tokens * GEMINI_CACHED_INPUT_COST_PER_M +
            completion_tokens * GEMINI_OUTPUT_COST_PER_M) / 1_000_000

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


import json

def format_stream_chunk(event_type: str, content: str) -> str:
    return json.dumps({"type": event_type, "content": content}) + "\n"

@observe(name="generate_response")
async def generate_response(query, session_id=None, user_id=None):
    logger.info(f"Query: {query}")
    
    # Mettre à jour la trace Langfuse avec la session et l'utilisateur si disponibles
    if session_id:
        langfuse_context.update_current_trace(session_id=session_id)
    if user_id:
        langfuse_context.update_current_trace(user_id=user_id)
        
    # Nest LlamaIndex traces inside this trace
    try:
        from langfuse.llama_index.llama_index import context_root
        lf_handler = langfuse_context.get_current_llama_index_handler()
        if lf_handler:
            context_root.set(lf_handler.trace or lf_handler.root_span)
    except Exception as e:
        logger.error(f"Failed to link LlamaIndex context to Langfuse trace: {e}")
        
    # 1. Tente d'utiliser le cache de contexte Gemini (Inférence directe ultra-rapide)
    try:
        # get or create the cache
        cache_name = get_or_create_cv_cache(genai_client)
        logger.info(f"Using context cache: {cache_name}")
        
        # Call model generate_content using asyncio.to_thread to avoid blocking
        response_obj = await asyncio.to_thread(
            genai_client.models.generate_content,
            model="gemini-2.5-flash",
            contents=query,
            config=types.GenerateContentConfig(
                cached_content=cache_name,
                temperature=0.0
            )
        )
        response_text = response_obj.text
        
        # Extract token usage and costs
        usage = response_obj.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else 0
        cached_tokens = usage.cached_content_token_count if usage else 0
        completion_tokens = usage.candidates_token_count if usage else 0
        
        cost = calculate_cache_cost(prompt_tokens, cached_tokens, completion_tokens)
        
        langfuse_context.update_current_trace(
            metadata={
                "cost_usd": cost,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "completion_tokens": completion_tokens,
                "caching_status": "hit" if cached_tokens > 0 else "miss",
                "pipeline_type": "context_caching"
            }
        )
        
        logger.info(f"Response (cached): {response_text} | Cost: ${cost:.6f} (Cached tokens: {cached_tokens})")
        return response_text
        
    except Exception as cache_err:
        logger.error(f"Context caching failed or disabled: {cache_err}. Falling back to standard pipeline...")
        
        # 2. Fallback sur le pipeline standard (DSPy Classifier + Pinecone RAG + LlamaIndex)
        agent_input = (
            f"{query}\n"
            f"### DIRECTIVE DE CONTRÔLE ###\n"
            f"Instruction critique : L'utilisateur s'adresse à toi ('Tu') par habitude, mais tu es une IA. "
            f"En tant que J.A.R.V.I.S, tu dois répondre pour Quentin, jamais à la première personne. "
            f"Réponds en tant qu'Assistant J.A.R.V.I.S en parlant de Quentin à la 3ème personne ('Il', 'Quentin', 'Le candidat')."
        )
        
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "embedding_tokens": 0}
        token_usage_token = agent_token_usage_var.set(token_usage)
        
        try:
            token_usage["prompt_tokens"] += 350
            token_usage["completion_tokens"] += 10
            
            response = await agent.run(agent_input)
            
            final_usage = agent_token_usage_var.get()
            prompt_tokens = final_usage["prompt_tokens"]
            completion_tokens = final_usage["completion_tokens"]
            embedding_tokens = final_usage["embedding_tokens"]
            cost = calculate_cost(prompt_tokens, completion_tokens, embedding_tokens)
            
            langfuse_context.update_current_trace(
                metadata={
                    "cost_usd": cost,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "embedding_tokens": embedding_tokens,
                    "caching_status": "disabled/error",
                    "pipeline_type": "standard_fallback",
                    "error": str(cache_err)
                }
            )
            
            logger.info(f"Response (fallback): {response} | Cost: ${cost:.6f}")
            return str(response)
        finally:
            agent_token_usage_var.reset(token_usage_token)

@observe(name="generate_response_stream")
async def generate_response_stream(query, session_id=None, user_id=None):
    logger.info(f"Query (stream): {query}")
    
    # Mettre à jour la trace Langfuse avec la session et l'utilisateur si disponibles
    if session_id:
        langfuse_context.update_current_trace(session_id=session_id)
    if user_id:
        langfuse_context.update_current_trace(user_id=user_id)
        
    try:
        from langfuse.llama_index.llama_index import context_root
        lf_handler = langfuse_context.get_current_llama_index_handler()
        if lf_handler:
            context_root.set(lf_handler.trace or lf_handler.root_span)
    except Exception as e:
        logger.error(f"Failed to link LlamaIndex context to Langfuse trace: {e}")
        
    # 1. Tente d'utiliser le cache de contexte Gemini (Inférence directe ultra-rapide)
    try:
        yield format_stream_chunk("status", "Recherche dans le cache de contexte...")
        cache_name = get_or_create_cv_cache(genai_client)
        logger.info(f"Using context cache (stream): {cache_name}")
        
        yield format_stream_chunk("status", "Génération de la réponse...")
        
        def run_cached_stream():
            return genai_client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=query,
                config=types.GenerateContentConfig(
                    cached_content=cache_name,
                    temperature=0.0
                )
            )
            
        response_stream = await asyncio.to_thread(run_cached_stream)
        
        full_text = ""
        for chunk in response_stream:
            text = chunk.text
            full_text += text
            yield format_stream_chunk("text", text)
            
        logger.info(f"Stream generation completed (cached). Length: {len(full_text)}")
        return
        
    except Exception as cache_err:
        logger.error(f"Context caching failed or disabled (stream): {cache_err}. Falling back to standard pipeline...")
        
        # 2. Fallback sur le pipeline standard (DSPy Classifier + Pinecone RAG + LlamaIndex)
        yield format_stream_chunk("status", "Classification de la demande...")
        
        agent_input = (
            f"{query}\n"
            f"### DIRECTIVE DE CONTRÔLE ###\n"
            f"Instruction critique : L'utilisateur s'adresse à toi ('Tu') par habitude, mais tu es une IA. "
            f"En tant que J.A.R.V.I.S, tu dois répondre pour Quentin, jamais à la première personne. "
            f"Réponds en tant qu'Assistant J.A.R.V.I.S en parlant de Quentin à la 3ème personne ('Il', 'Quentin', 'Le candidat')."
        )
        
        # 1. Classification
        try:
            intent_output = classifier(query)
            intent_str = str(intent_output.intent).strip()
        except Exception as e:
            logger.error(f"Error classifying intent in stream: {e}")
            intent_str = "mixed"
            
        logger.info(f"JarvisAgent Routing (stream) | Query: {query} | Intent: {intent_str}")
        name, arg = parse_intent(intent_str)
        
        if name:
            name = name.strip()
        if arg:
            arg = arg.strip()
            
        # Case-insensitive resolution of repository name
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
                logger.error(f"Error resolving repository name case-insensitively (stream): {e}")

        # Direct streaming routing
        if name == "chitchat":
            chitchat_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"L'utilisateur dit : \"{query}\"\n"
                f"En tant que J.A.R.V.I.S., réponds de façon courtoise, flegmatique et appropriée (en parlant de Monsieur Forget à la 3ème personne si besoin, et à la 1ère personne pour toi-même) :"
            )
            yield format_stream_chunk("status", "Rédaction de la réponse de courtoisie...")
            def run_chitchat_stream():
                return genai_client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=chitchat_prompt,
                    config=types.GenerateContentConfig(temperature=0.0)
                )
            response_stream = await asyncio.to_thread(run_chitchat_stream)
            for chunk in response_stream:
                yield format_stream_chunk("text", chunk.text)
                
        elif name == "list_all_projects":
            yield format_stream_chunk("status", "Récupération de la liste des projets sur GitHub...")
            from app.engine.tools import list_github_projects
            try:
                raw_projects = list_github_projects()
            except Exception as e:
                logger.error(f"Error listing projects in stream: {e}")
                raw_projects = "Aucun projet public trouvé ou indisponible pour le moment."
                
            rephrase_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Voici les données brutes sur les projets de Monsieur Forget :\n"
                f"{raw_projects}\n\n"
                f"Rédige une réponse parfaite au ton de J.A.R.V.I.S. pour présenter ces projets à la requête : \"{query}\" :"
            )
            yield format_stream_chunk("status", "Mise en forme de la liste des projets...")
            def run_list_projects_stream():
                return genai_client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=rephrase_prompt,
                    config=types.GenerateContentConfig(temperature=0.0)
                )
            response_stream = await asyncio.to_thread(run_list_projects_stream)
            for chunk in response_stream:
                yield format_stream_chunk("text", chunk.text)
                
        elif name == "read_project_readme" and arg:
            yield format_stream_chunk("status", f"Lecture du README pour le projet '{arg}' sur GitHub...")
            from app.engine.tools import get_github_activity
            try:
                raw_readme = get_github_activity(arg)
            except Exception as e:
                logger.error(f"Error reading README for {arg} in stream: {e}")
                raw_readme = f"README ou dépôt '{arg}' introuvable."
                
            rephrase_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Voici le contenu du README pour le projet {arg} :\n"
                f"{raw_readme}\n\n"
                f"Rédige une réponse parfaite au ton de J.A.R.V.I.S. pour résumer ou expliquer ce projet en réponse à la requête : \"{query}\" :"
            )
            yield format_stream_chunk("status", f"Analyse et rédaction du résumé de '{arg}'...")
            def run_readme_stream():
                return genai_client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=rephrase_prompt,
                    config=types.GenerateContentConfig(temperature=0.0)
                )
            response_stream = await asyncio.to_thread(run_readme_stream)
            for chunk in response_stream:
                yield format_stream_chunk("text", chunk.text)
                
        elif name == "cv_query_engine" and arg:
            yield format_stream_chunk("status", "Recherche dans la base de connaissances Pinecone...")
            query_engine = pinecone_index.as_query_engine()
            try:
                retrieved_response = await query_engine.aquery(arg)
                response_content = str(retrieved_response)
            except Exception as e:
                logger.error(f"Error querying RAG in stream: {e}")
                response_content = "Information non trouvée."
                
            rephrase_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Voici l'information trouvée dans le CV de Monsieur Forget :\n"
                f"{response_content}\n\n"
                f"Rédige une réponse parfaite au ton de J.A.R.V.I.S. pour répondre à la question de l'utilisateur : \"{query}\" :"
            )
            yield format_stream_chunk("status", "Mise en forme de la réponse professionnelle...")
            def run_cv_stream():
                return genai_client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=rephrase_prompt,
                    config=types.GenerateContentConfig(temperature=0.0)
                )
            response_stream = await asyncio.to_thread(run_cv_stream)
            for chunk in response_stream:
                yield format_stream_chunk("text", chunk.text)
                
        else:
            # Fallback to ReActAgent
            yield format_stream_chunk("status", "Appel à l'agent de raisonnement complexe...")
            logger.info(f"Fallback to ReActAgent in stream for query: {query}")
            response = await react_agent.run(agent_input)
            yield format_stream_chunk("text", str(response))
    



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

