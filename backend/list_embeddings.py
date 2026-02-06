import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    exit(1)

genai.configure(api_key=api_key)

print("Listing available Gemini embedding models:")
print("-" * 40)

found = False
try:
    for m in genai.list_models():
        if 'embed' in m.name or 'embedding' in m.name:
            found = True
            print(f"Name: {m.name}")
            print(f"Description: {m.description}")
            print(f"Input Token Limit: {m.input_token_limit}")
            print("-" * 40)
    
    if not found:
        print("No embedding models found.")

except Exception as e:
    print(f"An error occurred: {e}")
