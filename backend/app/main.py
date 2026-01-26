from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routers.chat import chat_router
import uvicorn
import os

app = FastAPI(
    title="CV Agent API",
    description="Backend pour l'agent IA de Quentin Forget",
    version="1.0.0"
)

# Configuration CORS (Pour autoriser ton Frontend à parler au Backend)
origins = [
    "http://localhost:3000", # Si tu utilises React/Next.js
    "http://localhost:8000",
    "https://cv-agent-two.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# On ajoute la route /api/chat à l'application
app.include_router(chat_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "L'API de l'Agent CV est en ligne 🚀"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
