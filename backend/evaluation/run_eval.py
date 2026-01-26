import os
import pandas as pd
from datasets import Dataset
import nest_asyncio
import asyncio
from ragas.run_config import RunConfig

# 1. Import LangChain's Gemini integrations
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.callbacks import BaseCallbackHandler

# 2. Import Ragas modules
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# 3. Import your agent
# Assurez-vous que l'import fonctionne (voir discussion précédente)
import sys
import os
# Import de vos outils (ceux que l'agent utilise)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.engine.generate import generate_response

# Appliquer le patch pour les boucles imbriquées (nécessaire pour Ragas + LlamaIndex)
nest_asyncio.apply()

# ------------------------------------------------------------------
# 🛠️ CLASSE DE DEBUG (POUR VOIR CE QUE VOIT GEMINI)
# ------------------------------------------------------------------
class GeminiDebugHandler(BaseCallbackHandler):
    """
    Cette classe intercepte les appels vers Gemini.
    Elle affiche le Prompt exact envoyé par Ragas et la réponse brute.
    """
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"\n\033[94m{'='*40} ENVOI À GEMINI (Prompt Ragas) {'='*40}\033[0m")
        for prompt in prompts:
            print(prompt)
        print(f"\033[94m{'='*100}\033[0m\n")

    def on_llm_end(self, response, **kwargs):
        print(f"\n\033[92m{'='*40} RÉPONSE DE GEMINI (Validation) {'='*40}\033[0m")
        # On affiche la première génération (souvent la seule)
        try:
            print(response.generations[0][0].text)
        except:
            print(response)
        print(f"\033[92m{'='*100}\033[0m\n")

# ------------------------------------------------------------------
# Step 1: Configuration
# ------------------------------------------------------------------
api_google = os.getenv("GOOGLE_API_KEY")

# Configuration Ragas pour éviter les limites de débit (Rate Limits)
my_run_config = RunConfig(
    max_workers=1,
    timeout=60,
    max_retries=2,
)

# Judge LLM (Gemini)
google_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Note: gemini-2.5 n'existe pas encore publiquement, utilisez 1.5
    api_key=api_google,
    generation_config={"temperature": 0},
    callbacks=[GeminiDebugHandler()]
)

# Embeddings
google_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# Wrappers Ragas
ragas_llm = LangchainLLMWrapper(google_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(google_embeddings)

# ------------------------------------------------------------------
# Step 2: Data Preparation
# ------------------------------------------------------------------
data_dict = {
    "question": [
        "Quelles sont tes compétences pour un poste de Machine Learning Engineer ou Data Scientist?",
        "Dans quelle entreprise as-tu travaillé en 2022 et peux-tu me donner des détails sur ta mission?",
        "Quelles sont tes diplomes ?",
        "Quel est ton diplôme le plus récent ?",
        "Peux-tu résumer ton dernier projet 'ai-cv' sur GitHub ?",
        "Quelles stacks techniques maitrise-tu ?",
        "Quels sont tes hobbies ? ",
        "Quelle est ta localisation ?",
        "Quelle est votre prétention salariale ?",
        "Quelle est votre disponibilité ?",
        "Quelles sont vos langues parlées ?"        
    ],
    "contexts": [
        [
            "Profil technique : Hybride Data Scientist et ML Engineer. Compétences en modélisation : Algorithmes classiques (Random Forest, régression logistique et linéaire) et Deep Learning (Réseaux de neurones MLP, CNN, RNN, Transformers). Compétences en Engineering (MLOps) : Conteneurisation des modèles avec Docker, création d'API pour la mise en production, au-delà des notebooks Jupyter."
        ],
        [
            "Expérience Professionnelle 2022 : Data Scientist chez Crédit Agricole (Équipe Workplace). Missions : Conception et automatisation de reporting (Excel, Python, Power BI). Analyse des données d’occupation (calculs de taux, tendances, segmentations). Recommandations d’optimisation pour typologies et capacités selon besoins."
        ],
        [
            "Formation et Éducation : Dans mon CV, dans la section formation, dans la sous section diplome, mes diplomes sont un Mastère en Data Science et Finance et une Licence en Mathématiques et Informatiques."
        ],
        [
            "Formation et Éducation : Dans mon CV, dans la section formation, dans la sous section diplome, Mon diplôme le plus récent est un Mastère en Data Science et Finance."
        ],
        [
            "Portfolio GitHub : Projet 'ai-cv'. Description : Assistant recrutement intelligent réinventant l'expérience candidat via l'IA Générative. Fonctionnalités : Transforme le CV statique en agent conversationnel dynamique capable de répondre aux recruteurs de manière contextuelle. Technologies utilisées : RAG (Retrieval Augmented Generation), Google Gemini."
        ],
        [
            "Dans mon CV, dans la section compétences techniques, Stack Technique - Langages : Python (Expert). Data Science : Pandas, NumPy, Scikit-learn. Deep Learning : Préférence pour TensorFlow/Keras (meilleure maîtrise que PyTorch). MLOps : En cours d'apprentissage de Docker et FastAPI pour la production. Intérêts actuels : IA Générative, intégration d'agents IA dans les applications web, frameworks d'évaluation et d'orchestration (LlamaIndex, LangChain, Ragas)."
        ],
        [
            "Centres d'intérêt et Loisirs : Dans mon CV, dans la section hobbies, Pratique régulière de la course en plein air, du padel et du tennis."
        ],
        [
            "Localisation : Dans mon CV, dans la section profil et la sous section localisation, ma localisation est en Ile de France"
        ],
        [
            "Prétention salariale : Dans le System Prompt, l'information est mentionnée que je recherche une rémunération de 45000 et 55000€/an"
        ],
        [
            "Disponibilité : Dans le System Prompt, l'information est mentionnée que je suis disponible dès maintenant"
        ],
        [
            "Langues : Dans mon CV, dans la section langues, Anglais (B2), Espagnol (A2)"
        ]
    ],
    "ground_truth": [
        "Je possède un profil hybride qui allie la rigueur mathématique du Data Scientist à la capacité de mise en production du ML Engineer. Concrètement, mes compétences se divisent en trois axes :La Modélisation : Je maîtrise les algorithmes machine learning comme Random Forest, régression logistique, régression linéaire, ainsi que le Deep Learning avec Réseaux de neurones MLP, CNN, RNN, Transformers. L'Engineering (MLOps) : Je ne m'arrête pas au Jupyter Notebook. Je sais conteneuriser mes modèles avec Docker et mettre en place l'API pour assurer la fiabilité en production.",
        "En 2022, j'étais en poste au sein du groupe Crédit Agricole. J'y occupais le role de Data Analyst au sein de l'équipe de Workplace, où j'ai pu travailler sur des projets pour la conception et automatisation de reporting (Excel, Python, Power BI), Analyse des données d’occupation - calculs de taux, tendances et segmentations, recommandations d’optimisation pour typologies et capacités selon besoins.",
        "Les diplomes que j'ai obtenu sont un Master en Data Science et Finance et une Licence en Mathématiques et Informatiques.",
        "Mon diplôme le plus récent est un Master en Data Science et Finance.",
        "Mon dernier projet sur GitHub est 'ai-cv'. Il s'agit d'un projet d'assistant recrutement intelligent. Une expérience candidat réinventée grâce à l'Intelligence Artificielle Générative. Ce projet transforme le CV statique en un agent conversationnel dynamique, utilisant le RAG (Retrieval Augmented Generation) et Google Gemini pour répondre aux recruteurs de manière contextuelle et personnalisée.",
        "Mon langage de prédilection est Python. Data Science : Pandas, NumPy, Scikit-learn. Deep Learning : J'ai une meilleure maitrise de TensorFlow/keras que de PyTorch. MLOps: Je suis en apprentissage de Docker et FastAPI pour la production de mes projets. Je suis aussi tourner vers les nouvelles tendances de l'IA Generative avec l'intégration des agents IA dans les applications web. Je me perfectionne dans les frameworks comme LlamaIndex, LangChain et Ragas pour l'évaluation de l'IA.",
        "Mes hobbies sont la course en pleine air, le padel et le tennis.",
        "Je suis basé en Ile de France",
        "Mon prétention salariale est de 45000 et 55000€/an",
        "Je suis disponible dès maintenant",
        "Je parle anglais (B2) et espagnol (A2)"
    ]
}

# ------------------------------------------------------------------
# Step 3: Generation Loop (CORRECTION MAJEURE)
# ------------------------------------------------------------------

async def generate_evaluate():
    """
    Fonction asynchrone pour générer les réponses de l'agent
    """
    print("⏳ Génération des réponses par l'agent en cours...")
    generated_answers = []
    
    for q in data_dict["question"]:
        # On injecte le contexte manuellement dans le prompt pour forcer l'agent à l'utiliser
        prompt = f"Answer this question: {q}\n"
        
        # Appel asynchrone
        response = await generate_response(prompt) 
        # OU await agent.chat(prompt) selon votre version de LlamaIndex
        
        # GESTION DE LA RÉPONSE (LlamaIndex vs LangChain)
        # LlamaIndex retourne souvent un objet avec .response, Langchain avec .content
        if hasattr(response, 'response'):
            answer_text = response.response
        elif hasattr(response, 'content'):
            answer_text = response.content
        if hasattr(response, 'blocks'):
            #answer_text = "\n".join([b.text for b in response.blocks if hasattr(b, 'text')])
            answer_text = response.blocks
        else:
            answer_text = str(response)
            
        generated_answers.append(answer_text)
        print(f"✅ Réponse générée pour : {q[:30]}...")
        
    return generated_answers

# Exécution de la boucle asynchrone
if __name__ == "__main__":
    
    # 1. On lance la génération (Ceci crée la boucle d'événement)
    answers = asyncio.run(generate_evaluate())
    
    # 2. On ajoute les réponses au dictionnaire
    data_dict["answer"] = answers

    # 3. Création du Dataset HuggingFace
    dataset = Dataset.from_dict(data_dict)

    # ------------------------------------------------------------------
    # Step 4: Run Evaluation
    # ------------------------------------------------------------------
    print("📊 Lancement de l'évaluation Ragas...")
    
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=my_run_config
    )

    # ------------------------------------------------------------------
    # Step 5: View Results
    # ------------------------------------------------------------------
    df_results = results.to_pandas()
    print("\nrésultats de l'évaluation :")
    print(df_results)
    
    # Optionnel : Sauvegarder en CSV
    df_results.to_csv("evaluation_results.csv")