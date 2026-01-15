from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.services.agent import chat_service
from pydantic import BaseModel
from app.core.logging import setup_logging


# Setup logging
setup_logging()

app = FastAPI(title="CV Agent API", description="API pour interagir avec l'agent CV Agent", version="1.0.0")

# 1. CONFIGURATION CORS (Autoriser le Frontend)
origins = [
    "http://localhost:3000", # L'adresse de votre Frontend Next.js
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Autorise POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)

# Modèle de données pour la requête
class ChatRequest(BaseModel):
    query: str

# 2. DÉFINITION DE LA ROUTE (L'adresse exacte)
# Notez le préfixe "/api" ici. C'est important.
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # On appelle votre agent intelligent
    response = await chat_service(request.query)
    return {"response": response}

# Route de test simple pour voir si le serveur est vivant
@app.get("/")
def read_root():
    return {"status": "CV Agent Backend is running"}
