import os
import uuid
import logging
from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from core.llm import get_llm
from core.vectorstore.loader import initialize_vectorstore, COLLECTION_NAMES
from core.agents.router_agent import create_router_agent

# Importar el router de chat
from api.routes.chat import router as chat_router

# Configuración de directorios
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Ajuste para subir un nivel más
PDF_DIRECTORY = os.path.join(BASE_DIR, "data", "raw", "pdf_docs")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "data", "processed", "vectorstores")
TEMPLATES_DIR = os.path.join(BASE_DIR, "src", "api", "templates")  # Ruta correcta a templates
STATIC_DIR = os.path.join(BASE_DIR, "src", "api", "static")      # Ruta correcta a static

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorstoreInitializationError(Exception):
    """Excepción personalizada para errores de inicialización del vectorstore."""
    pass

def initialize_system():
    """Inicializa los componentes básicos del sistema."""
    logger.info("Inicializando Sistema Multi-Agente Educativo...")
    logger.info("⚙️ Inicializando componentes...")
    try:
        # Inicializar LLM
        llm = get_llm()
        logger.info("✅ LLM inicializado")

        # Asegurar que existan los directorios necesarios
        os.makedirs(PDF_DIRECTORY, exist_ok=True)
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

        # Verificar la estructura de carpetas para las categorías
        for category in COLLECTION_NAMES:
            category_path = os.path.join(PDF_DIRECTORY, category)
            if not os.path.exists(category_path):
                logger.warning(f"⚠️ La carpeta de la categoría '{category}' no existe en {PDF_DIRECTORY}.  "
                               f"Asegúrate de que la estructura de carpetas sea correcta.")
                # Podrías crear la carpeta aquí si quieres:
                # os.makedirs(category_path, exist_ok=True)

        # Inicializar vectorstores
        vectorstores = initialize_vectorstore(
            pdf_directory=PDF_DIRECTORY,
            persist_directory=PERSIST_DIRECTORY
        )

        if not vectorstores:
            raise VectorstoreInitializationError("No se pudo inicializar ninguna colección de vectorstore")

        # Mostrar resumen de colecciones (mejorado)
        logger.info("📊 Colecciones disponibles:")
        for category, vs in vectorstores.items():
            try:
                collection_size = len(vs.get()['ids'])
                logger.info(f"   - {category}: {collection_size} documentos")
            except Exception as e:
                logger.error(f"Error al obtener el tamaño de la colección {category}: {e}")
                raise VectorstoreInitializationError(f"Error al acceder a la colección {category}")

        return llm, vectorstores, logger

    except VectorstoreInitializationError as e:
        logger.error(f"Error crítico de inicialización: {e}")
        raise  # Relanzar la excepción para que se maneje en el nivel superior
    except Exception as e:
        logger.exception(f"Error fatal al inicializar el sistema: {e}")  # Usar logger.exception
        raise  # Relanzar la excepción

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
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Configurar Jinja2Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

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
        response = router_agent(pregunta, {})  # Pasar un diccionario vacío como session_state
        return {"respuesta": response}

    except VectorstoreInitializationError as e:
        logger.error(f"Error de inicialización del vectorstore: {e}")
        raise HTTPException(status_code=500, detail=str(e))  # Devolver el mensaje de la excepción
    except Exception as e:
        logger.exception(f"Error en /consultar: {e}") # Usar logger.exception
        raise HTTPException(status_code=500, detail="Error interno del servidor")

# Incluir el router de chat (si lo tienes)
app.include_router(chat_router)