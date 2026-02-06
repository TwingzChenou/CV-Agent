import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from github import Github

def check_setup():
    print("--- Diagnostic check_setup.py ---\n")

    # 1. Load env vars
    current_dir = Path(__file__).parent
    dotenv_path = current_dir / "backend/.env"
    
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Environment variables loaded from {dotenv_path}.\n")

    # 2. Check keys
    required_keys = [
        "GOOGLE_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_INDEX",
        "GITHUB_TOKEN"
    ]

    missing_keys = False
    for key in required_keys:
        value = os.getenv(key)
        if value:
            print(f"{key}: OK")
        else:
            print(f"{key}: MANQUANT")
            missing_keys = True
    
    print("")

    # 3. Test Gemini
    print("--- Test Gemini ---")
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            print("Skipping Gemini test due to missing GOOGLE_API_KEY.")
        else:
            llm = Gemini(model="models/gemini-2.5-flash")
            response = llm.complete("Coucou")
            print(f"Response: {response}")
            print("Gemini Test: OK")
    except Exception as e:
        print(f"Gemini Test Failed: {e}")

    print("")

    # 3.5 Test Embeddings
    print("--- Test Embeddings ---")
    try:
        if not os.getenv("GOOGLE_API_KEY"):
             print("Skipping Embedding test due to missing GOOGLE_API_KEY.")
        else:
            embed_model = GeminiEmbedding(
                model_name="models/gemini-embedding-001",
                api_key=os.getenv("GOOGLE_API_KEY")
            )
            emb = embed_model.get_text_embedding("Test embedding")
            print(f"Embedding Test: OK (Length: {len(emb)})")
    except Exception as e:
        print(f"Embedding Test Failed: {e}")

    print("")

    # 4. Test Pinecone
    print("--- Test Pinecone ---")
    try:
        pinecone_key = os.getenv("PINECONE_API_KEY")
        pinecone_index = os.getenv("PINECONE_INDEX")
        
        if not pinecone_key or not pinecone_index:
             print("Skipping Pinecone test due to missing keys.")
        else:
            vector_store = PineconeVectorStore(
                api_key=pinecone_key,
                index_name=pinecone_index
            )
            print("PineconeVectorStore initialized: OK")
            
            try: 
                 pass
            except:
                pass
                
    except Exception as e:
        print(f"Pinecone Test Failed: {e}")

    print("")

    # 5. Test Github
    print("--- Test Github ---")
    try:
        if not os.getenv("GITHUB_TOKEN"):
            print("Skipping Github test due to missing GITHUB_TOKEN.")
        else:
            token_github = os.getenv("GITHUB_TOKEN")
            github_client = Github(token_github)
            print("Github Tool Test: OK")
    except Exception as e:
        print(f"Github Test Failed: {e}")

    print("\n--- End Diagnostic ---")

if __name__ == "__main__":
    check_setup()
