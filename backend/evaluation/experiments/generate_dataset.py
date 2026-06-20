import sys
import os
import asyncio
import json
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.tools import FunctionTool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.engine.tools import get_tools

async def main():
    print("🛠️ Chargement des outils de l'agent...")
    tools = get_tools()
    
    # On prépare le LLM "Créateur de scénarios"
    llm = Ollama(model="mistral", base_url="http://localhost:11434", request_timeout=120.0)
    embedding = OllamaEmbedding(model_name="mistral", base_url="http://localhost:11434")
    
    
    
    dataset = []

    print("🤖 Génération des scénarios basés sur les outils...")
    
    for tool in tools:
        # On récupère les infos de l'outil
        tool_name = tool.metadata.name
        

        # PROMPT : On demande à Ollama d'inventer des questions pour CET outil
        prompt = (
            f"Tu es un expert en test QA. Le but est de tester les outils de l'agent. Les questions porteront sur le CV et le profil de Quentin qui est un Data Scientist avec 2 ans d'expérience en Data Engineering. Les questions doivent etre en lien avec un entretien d'embauche. Tu dois poser des questions pour savoir si Quentin peut etre embauché. Voici un outil utilisé par un Agent IA.\n"
            f"Nom: {tool_name}\n"
            "Génère 10 questions utilisateurs variées (complexes, simples, directes) "
            "qui nécessiteraient impérativement d'utiliser cet outil.\n"
            "Format de réponse attendu : JSON pur (liste de strings)."
        )
        
        response = await llm.acomplete(prompt)
        
        # Nettoyage du JSON
        cleaned_json = response.text.replace("```json", "").replace("```", "").strip()
        
        try:
            questions = json.loads(cleaned_json)
            # On ajoute au dataset avec l'étiquette de l'outil attendu
            for q in questions:
                dataset.append({
                    "query": q,
                    "expected_tool": tool_name
                })
            print(f"✅ Questions générées pour l'outil : {tool_name}")
            
        except Exception as e:
            print(f"⚠️ Erreur de parsing pour l'outil {tool_name}: {e}")

    # Sauvegarde
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "datasets", "agent_RAG_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"🎉 Terminé ! {len(dataset)} scénarios sauvegardés dans {output_path}")

if __name__ == "__main__":
    asyncio.run(main())