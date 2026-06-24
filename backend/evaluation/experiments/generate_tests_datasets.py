import os
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader
from llama_index.llms.gemini import Gemini

# Initialisation des variables d'environnement et du sys.path
backend_root = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=backend_root / ".env")
sys.path.insert(0, str(backend_root))

from app.engine.tools import get_tools

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)

# --- SCHÉMAS PYDANTIC POUR LE FORMAT DE SORTIE ---
class ToolArgsModel(BaseModel):
    input: Optional[str] = Field(None, description="L'argument 'input' pour cv_query_engine (ex: 'Machine Learning', 'contacts', 'études').")
    repo: Optional[str] = Field(None, description="L'argument 'repo' pour read_project_readme (ex: 'CV-Agent', 'Momentum_AI').")

class ToolCallModel(BaseModel):
    name: str = Field(description="Nom exact de l'outil attendu")
    args: ToolArgsModel = Field(description="Arguments de l'appel d'outil")

class ScenarioModel(BaseModel):
    user_input: str = Field(description="Question ou ordre simulant un entretien d'embauche ou Quentin qui s'adresse à JARVIS")
    reference_tool_calls: list[ToolCallModel] = Field(description="Liste de l'outil ou des outils à activer")
    reference: str = Field(description="La réponse textuelle parfaite, complète et factuelle, basée STRICTEMENT sur le contexte, sans points de suspension.")

class BatchTestSuiteModel(BaseModel):
    scenarios: list[ScenarioModel] = Field(description="Liste des scénarios de test générés")


def resolve_refs(schema, defs=None):
    if defs is None:
        defs = schema.get("$defs", schema.get("definitions", {}))
    
    if isinstance(schema, dict):
        if "$ref" in schema:
            ref_path = schema["$ref"]
            ref_key = ref_path.split("/")[-1]
            ref_schema = defs[ref_key]
            return resolve_refs(ref_schema, defs)
        
        if "anyOf" in schema:
            non_null_types = [t for t in schema["anyOf"] if t.get("type") != "null"]
            if non_null_types:
                first_type = non_null_types[0]
                schema = {**schema, **first_type}
                schema.pop("anyOf", None)
            else:
                schema = {**schema, **schema["anyOf"][0]}
                schema.pop("anyOf", None)
        
        resolved = {}
        for k, v in schema.items():
            if k in ("$defs", "definitions", "additionalProperties", "title", "default"):
                continue
            resolved[k] = resolve_refs(v, defs)
        return resolved
        
    elif isinstance(schema, list):
        return [resolve_refs(item, defs) for item in schema]
        
    return schema


from app.engine.tools import get_tools, setup_gemini, setup_pinecone_index, list_github_projects, get_github_activity

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
"""


def generate_agent_test_suite(data_dir_path, output_path):
    logger.info("🚀 Démarrage de la génération dynamique du dataset de test (RAG + MCP)")
    
    # 1. Initialiser le LLM Gemini et l'index Pinecone (via configuration globale get_tools)
    local_llm = Gemini(model="models/gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
    schema_dict = resolve_refs(BatchTestSuiteModel.model_json_schema())
    
    # Configurer settings et obtenir l'index
    get_tools()
    embed_model = setup_gemini()
    pinecone_index = setup_pinecone_index(embed_model)
    query_engine = pinecone_index.as_query_engine()
    
    # 2. Charger les documents de contexte
    documents = SimpleDirectoryReader(input_dir=str(data_dir_path)).load_data()
    context_text = "\n\n".join([doc.text for doc in documents])
    
    # 3. Définition simplifiée et compacte des outils
    mcp_tools_definition = """
- Nom: "cv_query_engine"
  Description: Outil de recherche (RAG) pour répondre aux questions sur Quentin, son CV, son profil, ses compétences (NLP, Deep Learning, etc.), ses études, son expérience, ses contacts, ses hobbies, etc.
  Arguments attendus (JSON Schema): {"input": "sujet de la recherche"}
- Nom: "list_all_projects"
  Description: Outil pour lister tous les projets publics / dépôts GitHub de Quentin Forget.
  Arguments attendus (JSON Schema): {} (laisser vide)
- Nom: "read_project_readme"
  Description: Outil pour lire le README.md / la documentation d'un projet spécifique de Quentin (ex: "CV-Agent", "Momentum_AI", "business-saas").
  Arguments attendus (JSON Schema): {"repo": "nom_du_projet"}
"""

    all_questions = []
    batch_size = 5
    total_target = 100
    
    # Prompt de base réutilisable
    base_prompt = f"""
    Tu es un ingénieur QA expert en IA. Génère un dataset de test pour évaluer un Agent IA (RAG + MCP) nommé "JARVIS".
    JARVIS est l'assistant personnel de Quentin Forget.
    
    CONSIGNE CRITIQUE : Toutes les questions générées doivent porter UNIQUEMENT et STRICTEMENT sur les 3 sujets suivants liés à Quentin :
    1. Son CV et son Profil (RAG via 'cv_query_engine') : Son éducation/études, son expérience professionnelle, ses compétences en IA/NLP/Machine Learning/développement web, ses langues parlées, ses coordonnées de contact, ses disponibilités, ses hobbies/loisirs.
    2. Liste générale de ses projets (MCP via 'list_all_projects') : Demander à lister ses projets GitHub, voir son portfolio de projets, ou savoir ce qu'il a codé globalement.
    3. Détails d'un projet spécifique (MCP via 'read_project_readme') : Demander des détails, du code ou lire la documentation (README) d'un projet de Quentin (ex: "CV-Agent", "Momentum_AI", "business-saas").
    
    RÈGLES STRICTES DE CONTENU :
    - INTERDICTION de poser des questions générales non liées à Quentin (pas de questions sur la météo en général, comment coder un jeu d'échecs, des questions de mathématiques, des questions sur la gestion d'équipe générale, etc.).
    - Reste focalisé à 100% sur le contenu des documents fournis ci-dessous.
    
    EXEMPLES DE QUESTIONS ACCEPTABLES :
    - "JARVIS, quelles sont les compétences de Quentin en NLP ?" -> attend 'cv_query_engine' avec l'argument '{{"input": "NLP"}}'
    - "Peux-tu me lister les projets GitHub de Quentin ?" -> attend 'list_all_projects'
    - "Lis le README du projet CV-Agent" -> attend 'read_project_readme' avec l'argument '{{"repo": "CV-Agent"}}'
    - "Quentin a-t-il de l'expérience en Deep Learning ?" -> attend 'cv_query_engine' avec l'argument '{{"input": "Deep Learning"}}'
    - "Affiche mes coordonnées de contact" -> attend 'cv_query_engine' avec l'argument '{{"input": "contacts"}}'

    LISTE DES OUTILS DISPONIBLES :
    {mcp_tools_definition}
    
    CONTEXTE DES DOCUMENTS DE QUENTIN :
    {context_text}
    """

    # 4. Boucle de génération par paquets (Batching)
    batch_idx = 1
    total_batches = (total_target + batch_size - 1) // batch_size
    while len(all_questions) < total_target:
        current_needed = total_target - len(all_questions)
        current_batch_size = min(batch_size, current_needed)
        logger.info(f"⏳ Génération du Lot {batch_idx}/{total_batches} ({current_batch_size} questions)...")
        
        system_prompt = f"""
        {base_prompt}
        
        Génère exactement {current_batch_size} scénarios uniques et originaux portant STRICTEMENT sur le CV, le profil ou le GitHub de Quentin Forget. Varie les types de requêtes parmi les 3 catégories autorisées.
        
        Tu dois retourner obligatoirement un objet JSON contenant une liste de scénarios sous la clé "scenarios", structuré comme suit :
        {{
            "scenarios": [
                {{
                    "user_input": "Question ou requête de l'utilisateur",
                    "reference_tool_calls": [
                        {{
                            "name": "Nom_Exact_De_L_Outil",
                            "args": {{"nom_argument": "valeur"}}
                        }}
                    ],
                    "reference": "Réponse complète, rédigée et factuelle basée sur le contexte."
                }}
            ]
        }}
        """
        
        try:
            # On passe la structure Pydantic résolue à Gemini via 'generation_config'
            response = local_llm.complete(
                system_prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": schema_dict,
                }
            )
            batch_data = json.loads(response.text)
            
            # Extraction et sérialisation des lignes du lot
            for scenario in batch_data.get("scenarios", []):
                user_in = scenario.get("user_input", "").strip()
                tool_calls = scenario.get("reference_tool_calls", [])
                
                if not user_in or not tool_calls:
                    continue
                
                # Récupération dynamique du contexte selon les appels d'outils
                logger.info(f"   Génération de la référence parfaite pour '{user_in}'...")
                response_content = ""
                for tc in tool_calls:
                    name = tc.get("name")
                    args = tc.get("args", {})
                    if name == "cv_query_engine":
                        val = args.get("input")
                        if val:
                            try:
                                res = query_engine.query(val)
                                response_content += f"\n\nCONTEXTE DE RECHERCHE RAG ({val}) :\n{res}"
                            except Exception as err:
                                logger.error(f"Error querying RAG for ref: {err}")
                    elif name == "list_all_projects":
                        try:
                            response_content += f"\n\nDONNÉES DU PORTFOLIO (GITHUB PROJECTS) :\n{list_github_projects()}"
                        except Exception as err:
                            logger.error(f"Error listing github projects for ref: {err}")
                    elif name == "read_project_readme":
                        repo = args.get("repo")
                        if repo:
                            try:
                                response_content += f"\n\nCONTENU DU README POUR LE PROJET {repo} :\n{get_github_activity(repo)}"
                            except Exception as err:
                                logger.error(f"Error reading project readme for {repo} for ref: {err}")
                
                ref_prompt = f"""{SYSTEM_PROMPT}

Voici l'information issue des outils de Monsieur Forget :
{response_content if response_content else 'Aucune information trouvée.'}

REQUÊTE DE L'UTILISATEUR :
{user_in}

RÉPONSE PARFAITE AU TON DE J.A.R.V.I.S. (rédigée en français soutenu, flegmatique, parlant de Monsieur Forget à la 3ème personne et sans point de suspension) :"""
                
                try:
                    ref_response = local_llm.complete(ref_prompt)
                    ref = ref_response.text.strip()
                    logger.info(f"   -> Référence générée : {ref[:100]}...")
                except Exception as ref_err:
                    logger.error(f"   ❌ Échec de la génération de la référence : {ref_err}")
                    continue
                
                # Valider la référence générée
                if not ref or len(ref) < 15 or ref.endswith("?") or "..." in ref or "placeholder" in ref.lower():
                    logger.warning(f"   Scénario rejeté car la référence est invalide : {ref}")
                    continue
                    
                clean_tool_calls = []
                for tc in tool_calls:
                    clean_args = {k: v for k, v in tc.get("args", {}).items() if v is not None}
                    clean_tool_calls.append({
                        "name": tc.get("name"),
                        "args": clean_args
                    })
                
                all_questions.append({
                    "user_input": user_in,
                    "reference_tool_calls": clean_tool_calls,
                    "reference": ref
                })
                if len(all_questions) >= total_target:
                    break
                
            logger.info(f"✔️ Lot {batch_idx} traité. Total de questions valides : {len(all_questions)}/{total_target}")
            
        except Exception as e:
            logger.error(f"❌ Échec du lot {batch_idx} : {e}")
            batch_idx += 1
            continue
            
        batch_idx += 1

    # 5. Sauvegarde finale
    if not all_questions:
        logger.error("❌ Aucune question valide n'a pu être collectée.")
        sys.exit(1)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=4)
    logger.info(f"🎉 Génération terminée ! Fichier sauvegardé ({len(all_questions)} questions) : {output_path}")


# Point d'entrée de l'application
if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent.parent / "app" / "data"
    output_json_path = Path(__file__).resolve().parent.parent / "datasets/agent_test_suite_100.json"
    
    generate_agent_test_suite(data_dir, output_json_path)

