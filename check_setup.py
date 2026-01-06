import os
import sys
from dotenv import load_dotenv
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.pinecone import PineconeVectorStore

def check_setup():
    print("--- Diagnostic check_setup.py ---\n")

    # 1. Load env vars
    load_dotenv()
    print("Environment variables loaded.\n")

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
            llm = Gemini(model="models/gemini-2.5-flash") # Using a standard model for test
            response = llm.complete("Coucou")
            print(f"Response: {response}")
            print("Gemini Test: OK")
    except Exception as e:
        print(f"Gemini Test Failed: {e}")

    print("")

    # 4. Test Pinecone
    print("--- Test Pinecone ---")
    try:
        pinecone_key = os.getenv("PINECONE_API_KEY")
        pinecone_index = os.getenv("PINECONE_INDEX")
        
        if not pinecone_key or not pinecone_index:
             print("Skipping Pinecone test due to missing keys.")
        else:
            # Just trying to initialize the store to check if it connects/validates config
            # We are not inserting anything.
            # PineconeVectorStore usually connects lazily or on first call, 
            # but initializing it often validates the api key format or presence.
            # To be more sure, we might need to list indexes if using the raw client, 
            # but instructions say "Tente simplement d'initialiser le PineconeVectorStore"
            
            # Note: newer llama-index-vector-stores-pinecone might use 'pinecone_index' or 'index_name'
            # Let's try basic initialization.
            
            vector_store = PineconeVectorStore(
                api_key=pinecone_key,
                index_name=pinecone_index
            )
            print("PineconeVectorStore initialized: OK")
            
            # Optional: Try to access the index to verify connection if possible without heavy ops
            try: 
                 # This is specific to the underlying pinecone client usually wrapped.
                 # But the instruction says "Initialize ... to see if connection establishes".
                 # Initialization is often enough to check import and key presence.
                 pass
            except:
                pass
                
    except Exception as e:
        print(f"Pinecone Test Failed: {e}")


    # 5. Test Github
    print("--- Test Github ---")
    try:
        if not os.getenv("GITHUB_TOKEN"):
            print("Skipping Github test due to missing GITHUB_TOKEN.")
        else:
            github_client = GithubClient(github_token=GITHUB_TOKEN)
            print("Github Tool Test: OK")
    except Exception as e:
        print(f"Github Test Failed: {e}")

    print("\n--- End Diagnostic ---")

if __name__ == "__main__":
    check_setup()
