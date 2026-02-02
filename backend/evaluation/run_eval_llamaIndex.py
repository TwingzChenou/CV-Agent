import sys
import os
import json
import asyncio
import pandas as pd
import nest_asyncio
from dotenv import load_dotenv

# Patch asyncio
nest_asyncio.apply()

# --- IMPORTS LLAMAINDEX ---
from llama_index.core import Settings, Response
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.llms.gemini import Gemini
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.core.tools import QueryEngineTool, FunctionTool

# --- IMPORTS PROJET ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.engine.tools import get_tools # Ta fonction qui charge TOUS les outils

load_dotenv()

async def run_universal_evaluation():
    print("⚙️  Initialisation de l'évaluation universelle...")

    # 1. SETUP DU JUGE (Gemini Pro)
    judge_llm = Gemini(model="models/gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)
    
    # SETUP DU GÉNÉRATEUR (Pour simuler la réponse de l'agent sur les outils fonctionnels)
    generator_llm = Gemini(model="models/gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))

    faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
    relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)

    # 2. CHARGEMENT DES OUTILS
    # On récupère la liste réelle de tes outils (CV, GitHub, etc.)
    tools_list = get_tools()
    
    # On crée un dictionnaire pour les retrouver facilement par leur nom
    # { "cv_query_engine": <ToolObj>, "list_all_projects": <ToolObj> }
    tools_map = {t.metadata.name: t for t in tools_list}
    
    print(f"🔧 Outils chargés : {list(tools_map.keys())}")

    # 3. CHARGEMENT DU DATASET
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "datasets", "agent_RAG_dataset.json")
    if not os.path.exists(dataset_path):
        print("❌ Dataset introuvable.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # On prend tout le dataset (ou une partie pour tester)
    eval_data = raw_data # [:20] pour tester
    results = []

    print(f"🚀 Démarrage sur {len(eval_data)} scénarios...")

    for entry in eval_data:
        query = entry['query']
        tool_name = entry.get('expected_tool')

        # Si l'outil demandé n'existe pas dans notre configuration actuelle, on saute
        if tool_name not in tools_map:
            print(f"⚠️  Skip: Outil '{tool_name}' non trouvé dans get_tools()")
            continue

        print(f"\nTicketing ({tool_name}): '{query}'")
        current_tool = tools_map[tool_name]
        response_object = None

        try:
            # --- CAS 1 : C'EST UN MOTEUR DE RECHERCHE (RAG) ---
            if isinstance(current_tool, QueryEngineTool):
                # C'est facile, le moteur fait tout (Recherche + Réponse + Sources)
                response_object = await current_tool.query_engine.aquery(query)

            # --- CAS 2 : C'EST UNE FONCTION (GitHub) ---
            elif isinstance(current_tool, FunctionTool):
                # A. On exécute la fonction Python brute
                # Note: On suppose ici que la fonction ne prend pas d'arguments complexes
                # Pour 'list_github_projects', c'est bon. 
                # Pour 'read_readme', il faudrait extraire l'argument du query (plus complexe)
                
                tool_output = ""
                
                if current_tool.metadata.fn_schema:
                    # Si la fonction attend des arguments, on tente de les deviner (simplifié pour le test)
                    # Dans un vrai cas, il faudrait un Agent pour parser les arguments.
                    # Ici on teste surtout 'list_all_projects' qui n'a pas d'arguments.
                    try:
                        tool_output = current_tool.fn() 
                    except:
                        tool_output = "Erreur: Cette fonction nécessite des arguments que le script de test ne sait pas deviner."
                else:
                    tool_output = current_tool.fn()

                tool_output_str = str(tool_output)

                # B. On demande au LLM de formuler une réponse naturelle basée sur cet output
                # (On simule ce que l'Agent ferait après avoir reçu le retour de l'outil)
                prompt = (
                    f"Contexte fourni par l'outil technique : {tool_output_str}\n\n"
                    f"Question de l'utilisateur : {query}\n\n"
                    "Rédige une réponse claire et naturelle pour l'utilisateur basée UNIQUEMENT sur le contexte ci-dessus."
                )
                final_answer = await generator_llm.acomplete(prompt)

                # C. On CRÉE MANUELLEMENT un objet Response compatible avec l'Evaluateur
                # On met la sortie brute de l'outil comme "Source Node"
                fake_source_node = NodeWithScore(
                    node=TextNode(text=tool_output_str), 
                    score=1.0
                )
                
                response_object = Response(
                    response=final_answer.text,
                    source_nodes=[fake_source_node]
                )

            # --- ÉVALUATION (Commun aux deux cas) ---
            if response_object:
                # 1. Fidélité : Est-ce que la réponse colle aux sources (ou à la sortie de l'outil) ?
                eval_faith = await faithfulness_evaluator.aevaluate_response(response=response_object)
                
                # 2. Pertinence : Est-ce que ça répond à la question ?
                eval_rel = await relevancy_evaluator.aevaluate_response(query=query, response=response_object)

                score_faith = 1 if eval_faith.passing else 0
                score_rel = 1 if eval_rel.passing else 0

                print(f"  > Scores -> Fidélité: {score_faith} | Pertinence: {score_rel}")
                
                results.append({
                    "Outil": tool_name,
                    "Question": query,
                    "Réponse": response_object.response,
                    "Fidélité": score_faith,
                    "Pertinence": score_rel,
                    "Raison": eval_faith.feedback
                })

        except Exception as e:
            print(f"❌ Erreur sur {tool_name}: {e}")

    # 4. EXPORT DES RÉSULTATS
    if results:
        df = pd.DataFrame(results)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(current_dir, "datasets", "evaluation_results.csv")
        df.to_csv(output_file, index=False)
        
        # Moyennes par outil
        print("\n📊 MOYENNES PAR OUTIL :")
        print(df.groupby("Outil")[["Fidélité", "Pertinence"]].mean())
        
        print(f"\n💾 Résultats sauvegardés dans {output_file}")
    else:
        print("Aucun résultat généré.")

if __name__ == "__main__":
    asyncio.run(run_universal_evaluation())