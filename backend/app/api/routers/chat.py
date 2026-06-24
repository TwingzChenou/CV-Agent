from fastapi import APIRouter, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.engine.generate import generate_response, generate_response_stream
from app.core.limiter import limiter
import logging

# Configuration du router
chat_router = APIRouter()
logger = logging.getLogger("uvicorn")

# 1. Définir le format des données reçues (DTO)
class ChatRequest(BaseModel):
    message: str # L'utilisateur doit envoyer un JSON {"message": "Sa question"}
    session_id: str | None = None
    user_id: str | None = None
    stream: bool = False

class ChatResponse(BaseModel):
    response: str

# 2. Créer l'endpoint (L'URL sera /api/chat)
@chat_router.post("/chat")
@limiter.limit("15/minute")
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    """
    Endpoint pour discuter avec l'agent Quentin Forget.
    """
    try:
        user_message = chat_request.message
        logger.info(f"📩 Reçu API : {user_message} (stream={chat_request.stream})")

        if chat_request.stream:
            return StreamingResponse(
                generate_response_stream(
                    user_message,
                    session_id=chat_request.session_id,
                    user_id=chat_request.user_id
                ),
                media_type="text/event-stream"
            )
        else:
            # Appel à la logique métier standard
            ai_response = await generate_response(
                user_message,
                session_id=chat_request.session_id,
                user_id=chat_request.user_id
            )
            return ChatResponse(response=ai_response)

    except Exception as e:
        logger.error(f"❌ Erreur API : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )