#!/usr/bin/env python3
"""
Quick diagnostic script to verify Ollama is working and generating responses.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from llama_index.llms.ollama import Ollama
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.info("=" * 70)
logger.info("🔍 DIAGNOSTIC: Vérification de la connexion Ollama")
logger.info("=" * 70)

try:
    logger.info("📡 Initialisation du modèle Mistral...")
    llm = Ollama(model="mistral", request_timeout=60.0)
    logger.info("✅ Mistral initialisé")
    
    test_prompt = """Génère 3 questions simples et courtes (1 phrase chacune) pour tester un agent IA.
    Format: JSON array avec clés 'question' et 'expected_tool'.
    
    Réponds UNIQUEMENT avec le JSON, pas de texte avant ou après.
    
    Exemple:
    [
        {"question": "Quel est mon profil?", "expected_tool": "read_profile"},
        {"question": "Liste mes projets", "expected_tool": "list_projects"},
        {"question": "Crée un nouveau dossier", "expected_tool": "create_folder"}
    ]"""
    
    logger.info("🚀 Envoi d'une requête de test à Mistral (3 questions simples)...")
    logger.info("   ⏳ Attente de la réponse... (peut prendre 30-60 secondes)")
    
    start = datetime.now()
    response = llm.complete(test_prompt)
    elapsed = (datetime.now() - start).total_seconds()
    
    logger.info(f"✅ Réponse reçue après {elapsed:.1f}s")
    logger.info(f"📄 Contenu reçu ({len(response.text)} caractères):")
    logger.info("-" * 70)
    logger.info(response.text[:500])
    if len(response.text) > 500:
        logger.info(f"... (+ {len(response.text) - 500} caractères)")
    logger.info("-" * 70)
    
    logger.info("")
    logger.info("✅ Ollama fonctionne correctement!")
    logger.info("   📊 Vitesse: ~%d caractères/seconde" % (len(response.text) / elapsed))
    logger.info("")
    logger.info("🎯 Vous pouvez maintenant exécuter le script complet:")
    logger.info("   PYTHONPATH=backend python backend/evaluation/experiments/generate_tests_datasets.py")
    logger.info("")
    logger.info("   ⏱️  Temps estimé: 5-15 minutes pour générer 100 questions")
    logger.info("=" * 70)
    
except Exception as e:
    logger.error(f"❌ Erreur: {e}")
    logger.error("\nDiagnostic:")
    logger.error("  1. Ollama est-il en cours d'exécution? (ollama serve)")
    logger.error("  2. Le modèle Mistral est-il disponible? (ollama list)")
    logger.error("  3. Ollama écoute sur http://localhost:11434?")
    sys.exit(1)
