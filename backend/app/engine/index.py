import os
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from app.core.logging import setup_logging
import sys
import logging
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Connexion à Gemini et Pinecone
def connect_pinecone():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.Index(os.getenv("PINECONE_INDEX"))

# Setup storage context
def setup_storage_context(pc_index):
    return StorageContext.from_defaults(
    vector_store=PineconeVectorStore(pinecone_index=pc_index)
)

#setup embeddings
def setup_embeddings():
    return GeminiEmbedding(
        model_name="models/text-embedding-004",
        api_key=os.getenv("GOOGLE_API_KEY")
    )

#setup_clean_index
def clean_index(pc_index):
    return pc_index.delete(delete_all=True)


# Setup index
def setup_index(storage_context, documents):
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=setup_embeddings()
    )

def run_indexing_pipeline(documents):
    """
    This is the public function that loader.py will call to ingest cv documents.
    """
    logger.info("🚀 Starting index setup...")

    pc_index = connect_pinecone()
    logger.info("✅ Connected to Pinecone.")

    storage_context = setup_storage_context(pc_index)
    logger.info("✅ Storage context setup completed.")
    """ 
    clean_index(pc_index)
    logger.info("✅ Index cleaned.")
    """
    setup_index(storage_context, documents)
    logger.info("✅ Ingest documents completed.")


def main():
    logger.info("🚀 Starting index setup...")

    pc_index = connect_pinecone()
    logger.info("✅ Connected to Pinecone.")

    storage_context = setup_storage_context(pc_index)
    logger.info("✅ Storage context setup completed.")

if __name__ == "__main__":
    sys.exit(main())