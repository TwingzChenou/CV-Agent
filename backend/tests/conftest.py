import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# 1. Injecter des variables d'environnement de test fictives
os.environ["GOOGLE_API_KEY"] = "mock-google-key"
os.environ["PINECONE_API_KEY"] = "mock-pinecone-key"
os.environ["PINECONE_INDEX"] = "mock-pinecone-index"
os.environ["GITHUB_TOKEN"] = "mock-github-token"
os.environ["LANGFUSE_PUBLIC_KEY"] = "mock-public-key"
os.environ["LANGFUSE_SECRET_KEY"] = "mock-secret-key"
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

# 2. Importer les classes de mock natives de LlamaIndex
from llama_index.core.llms import MockLLM
from llama_index.core.embeddings import MockEmbedding
from llama_index.core import VectorStoreIndex

mock_embed = MockEmbedding(embed_dim=1536)
mock_index = VectorStoreIndex([], embed_model=mock_embed)
mock_llm = MockLLM()
mock_github = MagicMock()

# Lancer les patchs session-wide
patch("app.engine.tools.setup_llm", return_value=mock_llm).start()
patch("app.engine.tools.setup_gemini", return_value=mock_embed).start()
patch("app.engine.tools.setup_pinecone_index", return_value=mock_index).start()
patch("app.engine.tools.get_github_client", return_value=mock_github).start()

# Mock DSPy et son compilateur pour éviter des validations réseaux au chargement du module
mock_lm = MagicMock()
patch("dspy.LM", return_value=mock_lm).start()

mock_compiled = MagicMock()
patch("dspy.teleprompt.LabeledFewShot.compile", return_value=mock_compiled).start()

# 3. Fixture pour le TestClient FastAPI
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)
