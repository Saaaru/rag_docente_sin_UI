from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from core.agents.router_agent import create_router_agent
from core.llm import get_llm
from core.vectorstore.loader import initialize_vectorstore
import uuid
import logging
import os

router = APIRouter()
templates = Jinja2Templates(directory="src/api/templates")

# Diccionario global para mantener el estado de las sesiones
session_states = {}

# Variable global para el router
global_router = None

@router.get("/chat", response_class=HTMLResponse)
async def get_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@router.post("/chat")
async def post_chat(payload: dict):
    # Se espera que el payload tenga 'message' y opcionalmente 'session_id'
    message = payload.get("message", "")
    session_id = payload.get("session_id", "")
    
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
    
    if session_id not in session_states:
        # Inicializar el estado de la sesión
        session_states[session_id] = {
            "pending_request": False,
            "last_query": "",
            "asignatura": None,
            "nivel": None,
            "mes": None,
            "tipo": None,
            "categorias": []
        }
    
    # Inicializar global_router si no existe
    global global_router
    if global_router is None:
        PDF_DIRECTORY = os.path.join("C:/Users/Dante/Desktop/rag_docente_sin_UI-1/data", "raw")
        PERSIST_DIRECTORY = os.path.join("C:/Users/Dante/Desktop/rag_docente_sin_UI-1/data", "processed/vectorstores")
        logger = logging.getLogger("chat")
        llm = get_llm()
        vectorstores = initialize_vectorstore(pdf_directory=PDF_DIRECTORY, persist_directory=PERSIST_DIRECTORY)
        global_router = create_router_agent(llm, vectorstores, logger, session_id)
    
    # Llamar al router con el mensaje del usuario y el estado de la sesión
    response_text = global_router(message, session_states[session_id])
    
    return JSONResponse({"session_id": session_id, "response": response_text})