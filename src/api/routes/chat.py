from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from core.router import get_router_agent  # Importamos desde core.router
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Configurar templates
templates = Jinja2Templates(directory="api/templates")

@router.post("/chat", response_class=HTMLResponse)
async def chat_with_agent(
    request: Request,
    user_message: str = Form(...),
    session_id: str = Form(...)
):
    """Chatea con el agente."""
    try:
        # Obtenemos las dependencias del estado de la aplicación
        from main import llm, vectorstores, logger as main_logger
        
        # Obtenemos el router agent
        router_agent = get_router_agent(session_id, llm, vectorstores, main_logger)
        
        # Procesamos el mensaje
        agent_response = router_agent(user_message, {})

        return templates.TemplateResponse(
            "components/chat_response.html",
            {"request": request, "agent_response": agent_response, "session_id": session_id},
        )

    except Exception as e:
        logger.error(f"Error en la interacción con el agente: {e}")
        raise HTTPException(status_code=500, detail=str(e))