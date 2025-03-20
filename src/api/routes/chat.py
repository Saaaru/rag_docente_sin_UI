from fastapi import APIRouter, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from src.core.router import get_router_agent, RouterAgent  # Importamos el tipo
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Configurar templates
templates = Jinja2Templates(directory="src/api/templates")  # Ruta completa

@router.post("/chat", response_class=HTMLResponse)
async def chat_with_agent(
    request: Request,
    user_message: str = Form(...),
    session_id: str = Form(...),
    router_agent: RouterAgent = Depends(get_router_agent)  # Inyectamos el agente
):
    """Chatea con el agente, ahora usando el router y dependencias."""
    try:
        agent_response = router_agent(user_message, {})

        return templates.TemplateResponse(
            "components/chat_response.html",
            {"request": request, "agent_response": agent_response, "session_id": session_id},
        )

    except Exception as e:
        logger.error(f"Error en /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset", response_class=JSONResponse)
async def reset_conversation(session_id: str = Form(...)):
    """Reinicia la conversación."""
    from core.router import reset_router_agent  # Importamos aquí
    if reset_router_agent(session_id):
        logger.info(f"Conversación reiniciada para session_id: {session_id}")
        return {"message": "Conversación reiniciada"}
    return {"message": "No se encontró la conversación"}