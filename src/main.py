import os
import uuid
import logging
from fastapi import FastAPI, Request, Form, HTTPException, Depends, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession
from core.vectorstore.loader import initialize_vectorstore
from core.agents.router_agent import create_router_agent
from config.paths import (
    RAW_DIR,
    PERSIST_DIRECTORY,
    SRC_DIR,
    CREDENTIALS_DIR
)
from config.model_config import EMBEDDING_MODEL_NAME
from core.llm import get_llm  # <- Importamos get_llm desde core.llm
from core.router import get_router_agent, reset_router_agent  # Importamos desde core.router

# Importar el router de chat (si lo tienes)
from api import chat_router

# Configuración de directorios usando paths.py
TEMPLATES_DIR = SRC_DIR / "api" / "templates"
STATIC_DIR = SRC_DIR / "api" / "static"

# Añade esto para debug
print(f"📂 RAW_DIR: {RAW_DIR}")
print(f"📂 PERSIST_DIRECTORY: {PERSIST_DIRECTORY}")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorstoreInitializationError(Exception):
    """Excepción personalizada para errores de inicialización del vectorstore."""
    pass

# --- Configuración de Vertex AI ---
import os
from pathlib import Path

# Obtener el ID del proyecto y la ruta de las credenciales
PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID")
if not PROJECT_ID:
    # Si no está en las variables de entorno, intentar obtenerlo del archivo de credenciales
    try:
        import json
        credentials_path = str(CREDENTIALS_DIR / "proyecto-docente-453715-b625fbe2c520.json")
        with open(credentials_path, 'r') as f:
            creds_data = json.load(f)
            PROJECT_ID = creds_data.get('project_id')
            # Establecer la variable de entorno
            os.environ["GOOGLE_PROJECT_ID"] = PROJECT_ID
    except Exception as e:
        logger.error(f"Error al leer el archivo de credenciales: {e}")
        raise RuntimeError("No se pudo obtener el PROJECT_ID")

# Establecer GOOGLE_APPLICATION_CREDENTIALS
credentials_path = str(CREDENTIALS_DIR / "proyecto-docente-453715-b625fbe2c520.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

LOCATION = "us-central1"

# Inicializar Vertex AI con el PROJECT_ID explícito
print(f"Inicializando Vertex AI con PROJECT_ID: {PROJECT_ID}")
print(f"Usando credenciales de: {credentials_path}")

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    credentials=None  # Usará GOOGLE_APPLICATION_CREDENTIALS automáticamente
)

# Añade esta verificación justo después de definir credentials_path
if not os.path.exists(credentials_path):
    raise FileNotFoundError(f"No se encontró el archivo de credenciales en: {credentials_path}")
if not os.access(credentials_path, os.R_OK):
    raise PermissionError(f"No se puede leer el archivo de credenciales en: {credentials_path}")

def start_chat_session(model):
    """Inicia y devuelve una sesión de chat."""
    try:
        logger.info("Iniciando sesión de chat...")
        chat_session = model.start_chat()
        logger.info("Sesión de chat iniciada exitosamente.")
        return chat_session
    except Exception as e:
        logger.error(f"Error al iniciar la sesión de chat: {e}")
        return None

def initialize_system():
    """Inicializa los componentes básicos del sistema."""
    print("\nInicializando Sistema Multi-Agente Educativo...")
    
    try:
        # Inicializar LLM
        llm = get_llm()
        if not llm:
            raise VectorstoreInitializationError("No se pudo inicializar el LLM")
        print("✅ LLM inicializado")

        # Inicializar vectorstores
        vectorstores = initialize_vectorstore()
        if not vectorstores:
            raise VectorstoreInitializationError(
                "No se pudo inicializar ninguna colección de vectorstore. " +
                "Verifica que existan PDFs en subdirectorios de data/raw"
            )

        # Mostrar resumen de colecciones
        print("\n📊 Colecciones disponibles:")
        for collection_name, vs in vectorstores.items():
            try:
                collection_size = len(vs.get()['ids'])
                print(f"   ✓ {collection_name}: {collection_size} documentos")
            except Exception as e:
                print(f"   ⚠️ {collection_name}: Error al obtener tamaño - {e}")

        return llm, vectorstores, logger

    except VectorstoreInitializationError as e:
        logger.error(f"Error crítico de inicialización: {e}")
        raise
    except Exception as e:
        logger.exception(f"Error fatal al inicializar el sistema: {e}")
        raise

# Inicializar componentes *fuera* de la función main (para que sean globales)
try:
    llm, vectorstores, logger = initialize_system()
except VectorstoreInitializationError:
    logger.critical("No se puede continuar sin un vectorstore válido. Saliendo.")
    exit(1)  # Salir del programa si no se puede inicializar el vectorstore
except Exception:
    logger.critical("Error fatal durante la inicialización. Saliendo.")
    exit(1)

# Instanciar FastAPI
app = FastAPI()

# Montar el directorio de archivos estáticos
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Configurar Jinja2Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --- Gestión del Router Agent ---
_router_agents = {}  # Diccionario para almacenar los agentes por thread_id

def get_router_agent(thread_id: str):
    """Obtiene o crea un router agent para un thread_id dado."""
    global _router_agents
    if thread_id not in _router_agents:
        logger.info(f"Creando nuevo router agent para thread_id: {thread_id}")
        _router_agents[thread_id] = create_router_agent(
            llm=llm,
            vectorstores=vectorstores,
            logger=logger,
            thread_id=thread_id
        )
    return _router_agents[thread_id]

# --- Rutas de la API ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Sirve la página principal (index.html)."""
    thread_id = request.cookies.get("thread_id")
    if not thread_id:
        thread_id = str(uuid.uuid4())[:8]
        response = templates.TemplateResponse("chat.html", {"request": request, "thread_id": thread_id})
        response.set_cookie("thread_id", thread_id)
        return response
    return templates.TemplateResponse("chat.html", {"request": request, "thread_id": thread_id})

@app.post("/consultar", response_class=JSONResponse)
async def consultar(pregunta: str = Form(...), thread_id: str = Form(...)):
    """Maneja las consultas desde la interfaz web."""
    try:
        router_agent = get_router_agent(thread_id)
        response = router_agent(pregunta, {})
        return {"respuesta": response}

    except Exception as e:
        logger.exception(f"Error en /consultar: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

# Incluir el router de chat
app.include_router(chat_router)

# --- Rutas para chatear (ya corregidas en la respuesta anterior) ---
@app.post("/chat", response_class=HTMLResponse)
async def chat_with_agent(
    request: Request,
    user_message: str = Form(...),
    file: UploadFile = File(None),
    session_id: str = Form(...)
):
    """
    Chatea con el agente, utilizando una sesión de chat de Vertex AI.
    """
    llm = get_llm()  # Obtenemos el modelo
    if llm is None:
        raise HTTPException(status_code=500, detail="Error al cargar el modelo de lenguaje.")

    # --- Gestión de la sesión de chat (y agentes) ---
    global _router_agents  # Accedemos a la variable global
    if session_id not in _router_agents:
        # Creamos el router agent *solo si no existe*
        _router_agents[session_id] = create_router_agent(
            llm=llm,
            vectorstores=vectorstores,
            logger=logger,
            thread_id=session_id  # ¡Importante pasar el thread_id!
        )
    router_agent = _router_agents[session_id]


    if file:
        # ... (lógica para procesar el archivo, si es necesario) ...
        #  Aquí podrías, por ejemplo, extraer texto del PDF y agregarlo al mensaje del usuario.
        pass  # Dejamos esto como un placeholder por ahora

    try:
        # --- Aquí usamos el router agent ---
        #  El router agent se encarga de decidir si se necesita un agente especializado
        #  y de llamarlo si es necesario.
        agent_response = router_agent(user_message, {})  # Pasamos un diccionario vacío como session_state

        return templates.TemplateResponse(
            "components/chat_response.html",
            {"request": request, "agent_response": agent_response, "session_id": session_id},
        )

    except Exception as e:
        logger.error(f"Error en la interacción con el agente: {e}")
        raise HTTPException(status_code=500, detail=f"Error en la interacción con el agente: {e}")


@app.post("/reset")
async def reset_conversation(session_id: str = Form(...)):
    """Reinicia la conversación."""
    if reset_router_agent(session_id):
        logger.info(f"Conversación reiniciada para session_id: {session_id}")
        return {"message": "Conversación reiniciada"}
    return {"message": "No se encontró la conversación"}