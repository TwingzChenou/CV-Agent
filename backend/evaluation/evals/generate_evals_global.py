import os
import sys
import json
import asyncio
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import nest_asyncio
from langfuse.decorators import observe, langfuse_context

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
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, CBEventType, EventPayload
from llama_index.core.callbacks.base import BaseCallbackHandler
from contextvars import ContextVar
from app.engine.generate import agent

# Context variable for tracking token usage per request/context
agent_token_usage_var = ContextVar("agent_token_usage", default=None)

# Global variables for tracking Ragas evaluation token usage
is_evaluating_ragas = False
global_eval_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "embedding_tokens": 0}

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
            
        # Debugging print to see what events and payloads are received
        # print(f"[DEBUG] event_type: {event_type}, payload_keys: {list(payload.keys()) if payload else []}")
        
        global is_evaluating_ragas, global_eval_token_usage
        if is_evaluating_ragas:
            token_usage = global_eval_token_usage
        else:
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

# Register the context token counting handler globally in LlamaIndex Settings
token_handler = ContextTokenCountingHandler()
if Settings.callback_manager is None:
    Settings.callback_manager = CallbackManager([token_handler])
else:
    Settings.callback_manager.add_handler(token_handler)

# Initialisation des modèles Gemini (Juge distant)
gemini_llm = Gemini(model="models/gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
gemini_embed = GeminiEmbedding(model_name="models/gemini-embedding-001", api_key=os.getenv("GOOGLE_API_KEY"))

# Resolve the target classes from the DeprecationHelper wrappers in newer Ragas versions
LlamaIndexLLMClass = LlamaIndexLLMWrapper.new_target if hasattr(LlamaIndexLLMWrapper, "new_target") else LlamaIndexLLMWrapper
LlamaIndexEmbeddingsClass = LlamaIndexEmbeddingsWrapper.new_target if hasattr(LlamaIndexEmbeddingsWrapper, "new_target") else LlamaIndexEmbeddingsWrapper

# Custom subclass to handle Gemini-specific arguments and token tracking in LlamaIndex
class GeminiLlamaIndexLLMWrapper(LlamaIndexLLMClass):
    def check_args(self, n, temperature, stop, callbacks):
        # Translate arguments for LlamaIndex Gemini LLM to avoid TypeError
        gen_config = {}
        if temperature is not None:
            gen_config["temperature"] = temperature
        if stop is not None:
            gen_config["stop_sequences"] = stop
        return {"generation_config": gen_config}

    def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        prompt_text = prompt.to_string()
        prompt_tokens = estimate_tokens(prompt_text)
        
        result = super().generate_text(prompt, n, temperature, stop, callbacks)
        
        completion_text = ""
        if result.generations and result.generations[0]:
            completion_text = result.generations[0][0].text
        completion_tokens = estimate_tokens(completion_text)
        
        global is_evaluating_ragas, global_eval_token_usage
        if is_evaluating_ragas:
            global_eval_token_usage["prompt_tokens"] += prompt_tokens
            global_eval_token_usage["completion_tokens"] += completion_tokens
            
        return result

    async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
        prompt_text = prompt.to_string()
        prompt_tokens = estimate_tokens(prompt_text)
        
        result = await super().agenerate_text(prompt, n, temperature, stop, callbacks)
        
        completion_text = ""
        if result.generations and result.generations[0]:
            completion_text = result.generations[0][0].text
        completion_tokens = estimate_tokens(completion_text)
        
        global is_evaluating_ragas, global_eval_token_usage
        if is_evaluating_ragas:
            global_eval_token_usage["prompt_tokens"] += prompt_tokens
            global_eval_token_usage["completion_tokens"] += completion_tokens
            
        return result

# Custom subclass to track embedding tokens during Ragas evaluation
class CustomLlamaIndexEmbeddingsWrapper(LlamaIndexEmbeddingsClass):
    def embed_query(self, text: str):
        tokens = estimate_tokens(text)
        global is_evaluating_ragas, global_eval_token_usage
        if is_evaluating_ragas:
            global_eval_token_usage["embedding_tokens"] += tokens
        return super().embed_query(text)

    def embed_documents(self, texts: list[str]):
        tokens = sum(estimate_tokens(t) for t in texts)
        global is_evaluating_ragas, global_eval_token_usage
        if is_evaluating_ragas:
            global_eval_token_usage["embedding_tokens"] += tokens
        return super().embed_documents(texts)

    async def aembed_query(self, text: str):
        tokens = estimate_tokens(text)
        global is_evaluating_ragas, global_eval_token_usage
        if is_evaluating_ragas:
            global_eval_token_usage["embedding_tokens"] += tokens
        return await super().aembed_query(text)

    async def aembed_documents(self, texts: list[str]):
        tokens = sum(estimate_tokens(t) for t in texts)
        global is_evaluating_ragas, global_eval_token_usage
        if is_evaluating_ragas:
            global_eval_token_usage["embedding_tokens"] += tokens
        return await super().aembed_documents(texts)

# Envelopper les modèles pour Ragas
ragas_judge_llm = GeminiLlamaIndexLLMWrapper(gemini_llm)
ragas_judge_embed = CustomLlamaIndexEmbeddingsWrapper(gemini_embed)


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
        "answer_relevancy",
        "agent_prompt_tokens",
        "agent_completion_tokens",
        "agent_embedding_tokens",
        "agent_cost",
        "eval_prompt_tokens",
        "eval_completion_tokens",
        "eval_embedding_tokens",
        "eval_cost",
        "total_cost"
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
        if "cost" in col:
            print(f"  • {col.replace('_', ' ').title()} :")
            print(f"    - Moyenne : ${mean_val:.6f}")
            print(f"    - Min     : ${min_val:.6f}")
            print(f"    - Max     : ${max_val:.6f}")
        elif "tokens" in col:
            print(f"  • {col.replace('_', ' ').title()} :")
            print(f"    - Moyenne : {mean_val:.1f}")
            print(f"    - Min     : {min_val:.0f}")
            print(f"    - Max     : {max_val:.0f}")
        else:
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
    
    @observe(name="evaluation_scenario")
    async def run_scenario(row, index):
        query = row["user_input"]
        
        # Associer la session de test dans Langfuse
        session_id = f"eval-100-scenarios"
        langfuse_context.update_current_trace(
            session_id=session_id,
            user_id="evaluation-runner",
            tags=["evaluation"]
        )
        trace_id = langfuse_context.get_current_trace_id()
        
        # Nest LlamaIndex traces inside this trace
        try:
            from langfuse.llama_index.llama_index import context_root
            lf_handler = langfuse_context.get_current_llama_index_handler()
            if lf_handler:
                context_root.set(lf_handler.trace or lf_handler.root_span)
        except Exception as e:
            print(f"Failed to link LlamaIndex context to Langfuse trace: {e}")
        
        agent_input = (
            f"{query}\n"
            f"### DIRECTIVE DE CONTRÔLE ###\n"
            f"Instruction critique : L'utilisateur s'adresse à toi ('Tu') par habitude, mais tu es une IA. "
            f"En tant que J.A.R.V.I.S, tu dois répondre pour Quentin, jamais à la première personne. "
            f"Réponds en tant qu'Assistant J.A.R.V.I.S en parlant de Quentin à la 3ème personne ('Il', 'Quentin', 'Le candidat')."
        )
        # Initialize context-local token counting
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "embedding_tokens": 0}
        token_usage_token = agent_token_usage_var.set(token_usage)
        
        try:
            # Estimate DSPy classifier tokens (prompt ~ 350, completion ~ 10)
            token_usage["prompt_tokens"] += 350
            token_usage["completion_tokens"] += 10
            
            async with sem:
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
                
                # Fetch usage details
                final_usage = agent_token_usage_var.get()
                prompt_tokens = final_usage["prompt_tokens"]
                completion_tokens = final_usage["completion_tokens"]
                embedding_tokens = final_usage["embedding_tokens"]
                cost = calculate_cost(prompt_tokens, completion_tokens, embedding_tokens)
                
                return {
                    "trace_id": trace_id,
                    "agent_prompt_tokens": prompt_tokens,
                    "agent_completion_tokens": completion_tokens,
                    "agent_embedding_tokens": embedding_tokens,
                    "agent_cost": cost,
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
            final_usage = agent_token_usage_var.get()
            prompt_tokens = final_usage["prompt_tokens"]
            completion_tokens = final_usage["completion_tokens"]
            embedding_tokens = final_usage["embedding_tokens"]
            cost = calculate_cost(prompt_tokens, completion_tokens, embedding_tokens)
            return {
                "trace_id": trace_id,
                "agent_prompt_tokens": prompt_tokens,
                "agent_completion_tokens": completion_tokens,
                "agent_embedding_tokens": embedding_tokens,
                "agent_cost": cost,
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
        finally:
            agent_token_usage_var.reset(token_usage_token)

    tasks = [run_scenario(row, i) for i, row in enumerate(scenarios)]
    results_raw = await asyncio.gather(*tasks)
    
    print("✅ Exécutions de l'agent terminées.")
    print("⚖️ Le juge local (Mistral) commence l'analyse et la notation de l'agent...")
    
    trace_ids = [r["trace_id"] for r in results_raw]
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

    # Initialize token counts for Ragas evaluation
    global is_evaluating_ragas, global_eval_token_usage
    global_eval_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "embedding_tokens": 0}
    is_evaluating_ragas = True
    
    try:
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
    finally:
        is_evaluating_ragas = False
        final_eval_usage = global_eval_token_usage.copy()
    
    # Extract agent token counts and costs
    agent_prompt_tokens = [r.get("agent_prompt_tokens", 0) for r in results_raw]
    agent_completion_tokens = [r.get("agent_completion_tokens", 0) for r in results_raw]
    agent_embedding_tokens = [r.get("agent_embedding_tokens", 0) for r in results_raw]
    agent_costs = [r.get("agent_cost", 0.0) for r in results_raw]
    
    # Calculate Ragas evaluation metrics per scenario
    num_scenarios = len(scenarios)
    eval_prompt_per_scenario = final_eval_usage["prompt_tokens"] / num_scenarios if num_scenarios > 0 else 0
    eval_completion_per_scenario = final_eval_usage["completion_tokens"] / num_scenarios if num_scenarios > 0 else 0
    eval_embedding_per_scenario = final_eval_usage["embedding_tokens"] / num_scenarios if num_scenarios > 0 else 0
    
    total_eval_cost = calculate_cost(
        final_eval_usage["prompt_tokens"], 
        final_eval_usage["completion_tokens"], 
        final_eval_usage["embedding_tokens"]
    )
    eval_cost_per_scenario = total_eval_cost / num_scenarios if num_scenarios > 0 else 0
    
    # Affichage du rapport final combiné
    df_agent = results_agent.to_pandas()
    df_rag = results_rag.to_pandas()
    
    score_cols_rag = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    df_results = pd.concat([df_agent, df_rag[score_cols_rag]], axis=1)
    
    # Add new cost/token columns to df_results
    df_results["agent_prompt_tokens"] = agent_prompt_tokens
    df_results["agent_completion_tokens"] = agent_completion_tokens
    df_results["agent_embedding_tokens"] = agent_embedding_tokens
    df_results["agent_cost"] = agent_costs
    
    df_results["eval_prompt_tokens"] = [eval_prompt_per_scenario] * num_scenarios
    df_results["eval_completion_tokens"] = [eval_completion_per_scenario] * num_scenarios
    df_results["eval_embedding_tokens"] = [eval_embedding_per_scenario] * num_scenarios
    df_results["eval_cost"] = [eval_cost_per_scenario] * num_scenarios
    
    df_results["total_cost"] = df_results["agent_cost"] + df_results["eval_cost"]
    
    print("\n📊 RÉSULTATS DE L'ÉVALUATION :")
    print(df_results)
    
    # Sauvegarde des résultats
    df_results.to_csv(str(REPORT_PATH), index=False)
    print(f"\n💾 Rapport d'évaluation sauvegardé dans {REPORT_PATH}")
    
    # Envoi des scores vers Langfuse
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse import Langfuse
            langfuse_client = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
            )
            print("\n📤 Envoi des scores de l'évaluation globale vers Langfuse...")
            for i, trace_id in enumerate(trace_ids):
                if not trace_id:
                    continue
                # Liste des métriques à pousser
                metrics_to_send = [
                    "tool_call_accuracy",
                    "agent_goal_accuracy",
                    "topic_adherence(mode=f1)",
                    "tool_call_f1",
                    "context_precision",
                    "context_recall",
                    "faithfulness",
                    "answer_relevancy",
                    "agent_prompt_tokens",
                    "agent_completion_tokens",
                    "agent_embedding_tokens",
                    "agent_cost",
                    "eval_prompt_tokens",
                    "eval_completion_tokens",
                    "eval_embedding_tokens",
                    "eval_cost",
                    "total_cost"
                ]
                for metric in metrics_to_send:
                    if metric in df_results.columns:
                        val = df_results.loc[i, metric]
                        if pd.notna(val):
                            try:
                                langfuse_client.score(
                                    trace_id=trace_id,
                                    name=metric,
                                    value=float(val)
                                )
                            except Exception as score_err:
                                pass
            langfuse_client.flush()
            print("✅ Envoi des scores d'évaluation terminé.")
        except Exception as lf_err:
            print(f"❌ Échec de la connexion à Langfuse pour l'envoi des scores: {lf_err}")
    
    # Affichage des statistiques du dataset d'évaluation
    print_dataset_statistics(scenarios)
    
    # Affichage des statistiques des métriques d'évaluation
    print_metrics_statistics(df_results)

if __name__ == "__main__":
    asyncio.run(evaluate_agent())