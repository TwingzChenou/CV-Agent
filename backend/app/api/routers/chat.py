from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from app.engine.generate import generate_response
import logging

# Configuration du router
chat_router = APIRouter()
logger = logging.getLogger("uvicorn")

# 1. Définir le format des données reçues (DTO)
class ChatRequest(BaseModel):
    message: str # L'utilisateur doit envoyer un JSON {"message": "Sa question"}
    session_id: str | None = None
    user_id: str | None = None

class ChatResponse(BaseModel):
    response: str

# 2. Créer l'endpoint (L'URL sera /api/chat)
@chat_router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint pour discuter avec l'agent Quentin Forget.
    """
    try:
        user_message = request.message
        logger.info(f"📩 Reçu API : {user_message}")

        # Appel à ta logique métier (DSPy + LlamaIndex)
        ai_response = await generate_response(
            user_message,
            session_id=request.session_id,
            user_id=request.user_id
        )
        
        return ChatResponse(response=ai_response)

    except Exception as e:
        logger.error(f"❌ Erreur API : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )