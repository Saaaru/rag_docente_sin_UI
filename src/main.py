import os
import uuid
import logging
from fastapi import FastAPI, Request, Form, HTTPException, Depends, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import vertexai
from src.core.vectorstore.loader import initialize_vectorstore  # <-- AHORA CON src.
from src.config.paths import (  # <-- AHORA CON src.
        RAW_DIR,
        PERSIST_DIRECTORY,
        SRC_DIR,
        CREDENTIALS_DIR
    )
from src.core.llm import get_llm  # <-- AHORA CON src.
from src.core.router import get_router_agent, RouterAgent, reset_router_agent  # <-- AHORA CON src.
from src.api.routes import chat as chat_router  # <-- AHORA CON src.

# Configuración de directorios
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

# --- Configuración de Vertex AI (simplificada) ---
def initialize_vertexai():
    """Inicializa Vertex AI de forma segura."""
    project_id = os.environ.get("GOOGLE_PROJECT_ID")
    credentials_path = str(CREDENTIALS_DIR / "proyecto-docente-453715-b625fbe2c520.json")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path  # Siempre establecer

    if not project_id:
        try:
            import json
            with open(credentials_path, 'r') as f:
                project_id = json.load(f).get('project_id')
            if not project_id:
                raise ValueError("project_id no encontrado en credenciales.")
            os.environ["GOOGLE_PROJECT_ID"] = project_id  # Establecer si se encuentra
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error al obtener project_id: {e}")
            raise

    if not os.path.exists(credentials_path) or not os.access(credentials_path, os.R_OK):
        raise FileNotFoundError(f"Error de credenciales en: {credentials_path}")

    vertexai.init(project=project_id, location="us-central1")
    logger.info(f"Vertex AI inicializado. Project ID: {project_id}")

def initialize_system():
        """Inicializa LLM, vectorstores y verifica directorios."""
        print("\nInicializando Sistema Multi-Agente Educativo...")

        try:
            # Verificar directorios necesarios
            if not os.path.exists(RAW_DIR):
                logger.error(f"No existe el directorio RAW_DIR: {RAW_DIR}")
                raise FileNotFoundError(f"Directorio RAW_DIR no encontrado: {RAW_DIR}")

            if not os.path.exists(PERSIST_DIRECTORY):
                os.makedirs(PERSIST_DIRECTORY)
                logger.info(f"Creado directorio PERSIST_DIRECTORY: {PERSIST_DIRECTORY}")

            # Inicializar LLM
            llm = get_llm()
            if not llm:
                raise RuntimeError("No se pudo inicializar el LLM")
            print("✅ LLM inicializado")

            # Inicializar vectorstores con manejo de errores mejorado
            vectorstores = initialize_vectorstore()  # Se llama, pero NO se guarda en el ámbito global
            if not vectorstores:
                raise RuntimeError(
                    "No se pudo inicializar ninguna colección. Verifica data/raw"
                )

            # Mostrar resumen detallado de colecciones
            print("\n📊 Colecciones disponibles:")
            for collection_name, vs in vectorstores.items():
                try:
                    collection_size = len(vs.get()['ids'])
                    print(f"   ✓ {collection_name}: {collection_size} documentos")
                except Exception as e:
                    print(f"   ⚠️ {collection_name}: Error al obtener tamaño - {e}")

            return llm, logger  # SOLO llm y logger

        except Exception as e:
            logger.exception(f"Error fatal al inicializar el sistema: {e}")
            raise
# Inicializar componentes *fuera* de la función main (para que sean globales)
try:
    initialize_vertexai()  # Inicializar Vertex AI *antes* que nada
    llm, logger = initialize_system()
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

# --- Gestión del Router Agent (simplificado) ---
# Usamos la función get_router_agent directamente como dependencia

# --- Rutas de la API ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Sirve la página principal (index.html)."""
    thread_id = request.cookies.get("thread_id") or str(uuid.uuid4())[:8]
    response = templates.TemplateResponse("chat.html", {"request": request, "thread_id": thread_id})
    response.set_cookie("thread_id", thread_id)
    return response


@app.post("/consultar", response_class=JSONResponse)
async def consultar(pregunta: str = Form(...), thread_id: str = Form(...), router_agent: RouterAgent = Depends(get_router_agent)):
    """Maneja las consultas (usando inyección de dependencias)."""
    try:
        response = router_agent(pregunta, {})
        return {"respuesta": response}
    except Exception as e:
        logger.exception(f"Error en /consultar: {e}")
        raise HTTPException(status_code=500, detail="Error interno")

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
    Chatea con el agente.
    """
    llm = get_llm()  # Obtenemos el modelo
    if llm is None:
        raise HTTPException(status_code=500, detail="Error al cargar el modelo de lenguaje.")

    # --- Gestión de la sesión de chat (y agentes) ---
    global _router_agents  # Accedemos a la variable global
    if session_id not in _router_agents:
        # Creamos el router agent *solo si no existe*
        _router_agents[session_id] = get_router_agent(thread_id=session_id)
        
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