from fastapi import APIRouter, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from src.core.router import get_router_agent, RouterAgent
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Configurar templates
templates = Jinja2Templates(directory="src/api/templates")

@router.post("/api/chat", response_class=HTMLResponse)
async def chat_with_agent(
    request: Request,
    user_message: str = Form(...),
    thread_id: str = Form(...),  # Unificamos a thread_id
    router_agent: RouterAgent = Depends(get_router_agent)
):
    """Chatea con el agente, ahora usando el router y dependencias."""
    # Logs para depuración
    logger.info(f"API router - Recibida solicitud de chat")
    logger.info(f"thread_id: {thread_id}")
    logger.info(f"user_message: {user_message}")
    
    try:
        agent_response = router_agent(user_message, {})
        logger.info(f"Respuesta generada exitosamente")
        
        return templates.TemplateResponse(
            "components/chat_response.html",
            {"request": request, "agent_response": agent_response, "thread_id": thread_id},
        )
    except Exception as e:
        logger.error(f"Error en /chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset", response_class=JSONResponse)
async def reset_conversation(thread_id: str = Form(...)):  # Unificamos a thread_id
    """Reinicia la conversación."""
    from src.core.router import reset_router_agent  # Corregir import
    if reset_router_agent(thread_id):
        logger.info(f"Conversación reiniciada para thread_id: {thread_id}")
        return {"message": "Conversación reiniciada"}
    return {"message": "No se encontró la conversación"}

@router.post("/debug-chat")
async def debug_chat(request: Request):
    """Endpoint temporal para depuración."""
    form = await request.form()
    return {
        "received_data": dict(form),
        "content_type": request.headers.get("content-type"),
        "method": request.method,
        "cookies": request.cookies
    }