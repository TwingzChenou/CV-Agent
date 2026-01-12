import json
import os
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
import nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()

def load_cv_data(file_path):
    print(f"Loading data from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}

def format_value(value):
    if isinstance(value, list):
        return ", ".join([str(v) for v in value])
    return str(value)

def create_documents(data):
    print("Creating documents...")
    documents = []
    
    # Iterate over top-level keys
    for category, content in data.items():
        text_content_parts = []
        
        if isinstance(content, dict):
            # Handle dictionary content (e.g., profil, competences_techniques)
            for key, value in content.items():
                text_content_parts.append(f"{key}: {format_value(value)}")
        elif isinstance(content, list):
            # Handle list content (e.g., chronologie_professionnelle, formation)
            for item in content:
                if isinstance(item, dict):
                    item_parts = []
                    for key, value in item.items():
                        item_parts.append(f"{key}: {format_value(value)}")
                    text_content_parts.append("\n".join(item_parts))
                    text_content_parts.append("---") # Separator for list items
                else:
                    text_content_parts.append(str(item))
        else:
            text_content_parts.append(str(content))
            
        text_content = "\n".join(text_content_parts)
        
        # Clean up trailing separators if any
        if text_content.endswith("---\n"):
            text_content = text_content[:-4]
            
        doc = Document(
            text=text_content,
            metadata={"category": category}
        )
        documents.append(doc)
        
    print(f"Created {len(documents)} documents.")
    return documents

def main():
    # Check for required environment variables
    required_vars = ["GOOGLE_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

    # 1. Load Data
    cv_data = load_cv_data("cv.json")
    if not cv_data:
        return

    # 2. Document Creation
    documents = create_documents(cv_data)
    
    # 3. Embeddings Configuration
    print("Configuring embeddings...")
    Settings.embed_model = GeminiEmbedding(
        model_name="models/text-embedding-004",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # 4. Vector Store Initialization
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index = pc.Index(os.getenv("PINECONE_INDEX"))

    print("Cleaning up old index data...")
    try:
        pinecone_index.delete(delete_all=True)
        print("Index cleaned.")
    except Exception as e:
        print(f"Warning during cleanup: {e}")
    
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 5. Indexing
    print("Indexing documents...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    
    print("Indexing completed successfully!")

if __name__ == "__main__":
    main()
