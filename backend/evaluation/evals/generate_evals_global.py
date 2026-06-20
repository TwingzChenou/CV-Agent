import os
import sys
import json
import asyncio
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import nest_asyncio

# Appliquer nest_asyncio pour éviter les conflits de boucles d'événements
nest_asyncio.apply()

# Résolution dynamique des chemins
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent.parent
DATASET_PATH = SCRIPT_DIR.parent / "datasets" / "agent_test_suite_100.json"
REPORT_PATH = SCRIPT_DIR.parent / "datasets" / "ragas_eval_report.csv"

# Configuration de l'environnement backend
load_dotenv(dotenv_path=BACKEND_ROOT / ".env")
sys.path.insert(0, str(BACKEND_ROOT))

# Imports Ragas (utilisation des classes héritant de Metric pour la compatibilité avec evaluate)
# pyrefly: ignore [missing-import]
from ragas.dataset_schema import EvaluationDataset
from ragas import evaluate
from ragas.metrics import (
    ToolCallAccuracy,
    AgentGoalAccuracyWithReference,
    _TopicAdherenceScore,
    ToolCallF1,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerRelevancy
)
from ragas.messages import HumanMessage, AIMessage, ToolCall
from ragas.llms.base import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.run_config import RunConfig

# Imports LlamaIndex & Agent
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from app.engine.generate import agent

# Initialisation des modèles Gemini (Juge distant)
gemini_llm = Gemini(model="models/gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
gemini_embed = GeminiEmbedding(model_name="models/gemini-embedding-001", api_key=os.getenv("GOOGLE_API_KEY"))

# Custom subclass to handle Gemini-specific arguments in LlamaIndex
class GeminiLlamaIndexLLMWrapper(LlamaIndexLLMWrapper):
    def check_args(self, n, temperature, stop, callbacks):
        # Translate arguments for LlamaIndex Gemini LLM to avoid TypeError
        gen_config = {}
        if temperature is not None:
            gen_config["temperature"] = temperature
        if stop is not None:
            gen_config["stop_sequences"] = stop
        return {"generation_config": gen_config}

# Envelopper les modèles pour Ragas
ragas_judge_llm = GeminiLlamaIndexLLMWrapper(gemini_llm)
ragas_judge_embed = LlamaIndexEmbeddingsWrapper(gemini_embed)


run_config = RunConfig(
    timeout=180,       # 3 minutes timeout
    max_workers=4,     # more concurrent workers for Gemini API
    max_retries=10,
    max_wait=60
)

REFERENCE_TOPICS = [
    "Quentin Forget",
    "CV",
    "Projets GitHub",
    "Expériences professionnelles",
    "Études et formations",
    "Compétences techniques",
    "Hobbies et loisirs",
    "Coordonnées de contact"
]

def print_dataset_statistics(scenarios):
    print("\n📊 STATISTIQUES DU DATASET D'ÉVALUATION :")
    total_scenarios = len(scenarios)
    print(f"  • Nombre total de scénarios : {total_scenarios}")
    
    tool_counts = {}
    total_tool_calls = 0
    multi_tool_scenarios = 0
    for s in scenarios:
        tcs = s.get("reference_tool_calls", [])
        if len(tcs) > 1:
            multi_tool_scenarios += 1
        for tc in tcs:
            tool_name = tc.get("name", "Unknown")
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            total_tool_calls += 1
            
    print(f"  • Nombre total d'appels d'outils référencés : {total_tool_calls}")
    print(f"  • Scénarios multi-outils : {multi_tool_scenarios}")
    print("  • Distribution des outils de référence :")
    for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_scenarios) * 100 if total_scenarios > 0 else 0
        print(f"    - {tool}: {count} ({percentage:.1f}%)")
        
    avg_input_len = sum(len(s.get("user_input", "")) for s in scenarios) / total_scenarios if total_scenarios > 0 else 0
    avg_ref_len = sum(len(s.get("reference", "")) for s in scenarios) / total_scenarios if total_scenarios > 0 else 0
    print(f"  • Longueur moyenne des requêtes utilisateur : {avg_input_len:.1f} caractères")
    print(f"  • Longueur moyenne des réponses de référence : {avg_ref_len:.1f} caractères")
    print("-" * 50)

def print_metrics_statistics(df_results):
    print("\n📈 STATISTIQUES DES MÉTRIQUES D'ÉVALUATION (Moyenne, Min, Max) :")
    score_cols = [
        "tool_call_accuracy",
        "agent_goal_accuracy",
        "topic_adherence(mode=f1)",
        "tool_call_f1",
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy"
    ]
    # Filter for columns that actually exist in the dataframe
    existing_cols = [col for col in score_cols if col in df_results.columns]
    
    if not existing_cols:
        print("  Aucune colonne de score d'évaluation trouvée dans les résultats.")
        return
        
    for col in existing_cols:
        series = df_results[col].dropna()
        if series.empty:
            print(f"  • {col} : Pas de données valides")
            continue
        mean_val = series.mean()
        min_val = series.min()
        max_val = series.max()
        print(f"  • {col.replace('_', ' ').title()} :")
        print(f"    - Moyenne : {mean_val:.3f}")
        print(f"    - Min     : {min_val:.3f}")
        print(f"    - Max     : {max_val:.3f}")
    print("-" * 50)

async def evaluate_agent():
    # 1. Chargement du dataset
    if not DATASET_PATH.exists():
        print(f"❌ Dataset de test introuvable à l'adresse : {DATASET_PATH}")
        sys.exit(1)
        
    print(f"📊 Chargement du dataset depuis {DATASET_PATH}...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
            scenarios = scenarios[:limit]
            print(f"⚠️ Limité à {limit} scénarios pour ce test.")
        except ValueError:
            pass
    
    print(f"🚀 Exécution de l'agent sur {len(scenarios)} scénarios...")
    
    # Utilisation d'un sémaphore pour ne pas surcharger les APIs Gemini (agent)
    sem = asyncio.Semaphore(5)
    
    async def run_scenario(row):
        query = row["user_input"]
        agent_input = (
            f"{query}\n"
            f"### DIRECTIVE DE CONTRÔLE ###\n"
            f"Instruction critique : L'utilisateur s'adresse à toi ('Tu') par habitude, mais tu es une IA. "
            f"En tant que J.A.R.V.I.S, tu dois répondre pour Quentin, jamais à la première personne. "
            f"Réponds en tant qu'Assistant J.A.R.V.I.S en parlant de Quentin à la 3ème personne ('Il', 'Quentin', 'Le candidat')."
        )
        async with sem:
            try:
                response = await agent.run(agent_input)
                
                # Extraction des appels d'outils réels de l'agent
                ragas_tool_calls = [
                    ToolCall(name=getattr(tc, 'tool_name', ''), args=getattr(tc, 'tool_kwargs', {}))
                    for tc in getattr(response, "tool_calls", [])
                ]
                
                # Extraction de la réponse textuelle
                response_text = response.response.content if hasattr(response.response, 'content') else str(response.response)
                
                # Conversion des appels d'outils de référence
                reference_tool_calls = [
                    ToolCall(name=tc['name'], args=tc.get('args', {}))
                    for tc in row["reference_tool_calls"]
                ]
                
                # Extraction des contextes récupérés par le RAG
                retrieved_contexts = [
                    ns.node.get_content() if hasattr(ns.node, 'get_content') else getattr(ns.node, 'text', '')
                    for ns in getattr(response, "source_nodes", [])
                    if hasattr(ns, 'node')
                ]
                
                return {
                    "multi_turn": {
                        "user_input": [
                            HumanMessage(content=query),
                            AIMessage(content=response_text, tool_calls=ragas_tool_calls)
                        ],
                        "reference_tool_calls": reference_tool_calls,
                        "reference": row["reference"],
                        "reference_topics": REFERENCE_TOPICS
                    },
                    "single_turn": {
                        "user_input": query,
                        "response": response_text,
                        "reference": row["reference"],
                        "retrieved_contexts": retrieved_contexts
                    }
                }
            except Exception as e:
                print(f"❌ Erreur lors de l'exécution de la requête '{query}': {e}")
                return {
                    "multi_turn": {
                        "user_input": [HumanMessage(content=query), AIMessage(content="Error", tool_calls=[])],
                        "reference_tool_calls": [],
                        "reference": row["reference"],
                        "reference_topics": REFERENCE_TOPICS
                    },
                    "single_turn": {
                        "user_input": query,
                        "response": "Error",
                        "reference": row["reference"],
                        "retrieved_contexts": []
                    }
                }

    tasks = [run_scenario(row) for row in scenarios]
    results_raw = await asyncio.gather(*tasks)
    
    print("✅ Exécutions de l'agent terminées.")
    print("⚖️ Le juge local (Mistral) commence l'analyse et la notation de l'agent...")
    
    samples_multi = [r["multi_turn"] for r in results_raw]
    samples_single = [r["single_turn"] for r in results_raw]
    
    dataset_multi = EvaluationDataset.from_list(samples_multi)
    dataset_single = EvaluationDataset.from_list(samples_single)
    
    agent_goal_accuracy_metric = AgentGoalAccuracyWithReference()
    agent_goal_accuracy_metric.workflow_prompt.instruction = (
        "Given an agentic workflow comprised of Human, AI and Tools, identify the user_goal (the task or objective the user wants to achieve) "
        "and the end_state (a comprehensive, detailed summary of the final factual information and answers provided by the AI, "
        "preserving key technical and personal details)."
    )
    agent_goal_accuracy_metric.compare_outcome_prompt.instruction = (
        "Given user goal, desired outcome and achieved outcome, compare them and identify if they are semantically matching (1) or different (0). "
        "The verdict should be 1 if the achieved outcome successfully answers the user goal and aligns with the core facts of the desired outcome, "
        "even if the achieved outcome is more concise or omits some extra details present in the desired outcome. "
        "Only output 0 if there is a clear factual contradiction or if the core answer/goal is missed."
    )

    print("⚖️ 1. Lancement de l'évaluation des métriques d'Agent (Multi-turn)...")
    results_agent = evaluate(
        dataset=dataset_multi,
        metrics=[
            ToolCallAccuracy(),
            agent_goal_accuracy_metric,
            _TopicAdherenceScore(),
            ToolCallF1()
        ],
        llm=ragas_judge_llm,
        embeddings=ragas_judge_embed,
        run_config=run_config
    )
    
    print("⚖️ 2. Lancement de l'évaluation des métriques RAG (Single-turn)...")
    results_rag = evaluate(
        dataset=dataset_single,
        metrics=[
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy()
        ],
        llm=ragas_judge_llm,
        embeddings=ragas_judge_embed,
        run_config=run_config
    )
    
    # Affichage du rapport final combiné
    df_agent = results_agent.to_pandas()
    df_rag = results_rag.to_pandas()
    
    score_cols_rag = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    df_results = pd.concat([df_agent, df_rag[score_cols_rag]], axis=1)
    
    print("\n📊 RÉSULTATS DE L'ÉVALUATION :")
    print(df_results)
    
    # Sauvegarde des résultats
    df_results.to_csv(str(REPORT_PATH), index=False)
    print(f"\n💾 Rapport d'évaluation sauvegardé dans {REPORT_PATH}")
    
    # Affichage des statistiques du dataset d'évaluation
    print_dataset_statistics(scenarios)
    
    # Affichage des statistiques des métriques d'évaluation
    print_metrics_statistics(df_results)

if __name__ == "__main__":
    asyncio.run(evaluate_agent())