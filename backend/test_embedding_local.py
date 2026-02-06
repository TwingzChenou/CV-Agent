import os
import sys
from dotenv import load_dotenv
from llama_index.embeddings.gemini import GeminiEmbedding

# Load env
load_dotenv(dotenv_path="backend/.env")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("No GOOGLE_API_KEY found.")
    sys.exit(1)

def test_embedding(model_name):
    print(f"\nTesting model: {model_name}")
    try:
        embed_model = GeminiEmbedding(
            model_name=model_name,
            api_key=api_key
        )
        # Try to get a text embedding
        emb = embed_model.get_text_embedding("Hello world")
        print(f"Success! Embedding length: {len(emb)}")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

if __name__ == "__main__":
    # Test without prefix
    res1 = test_embedding("gemini-embedding-001")
    
    # Test with prefix
    res2 = test_embedding("models/gemini-embedding-001")
    
    if res1 and res2:
        print("\nBoth formats work locally.")
    elif res1:
        print("\nOnly 'gemini-embedding-001' works.")
    elif res2:
        print("\nOnly 'models/gemini-embedding-001' works.")
    else:
        print("\nNeither works.")
