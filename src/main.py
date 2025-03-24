import os
import uuid
import logging
from fastapi import FastAPI, Request, Form, HTTPException, Depends, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import vertexai
from src.core.vectorstore.loader import initialize_vectorstore
from src.core.vectorstore.retriever import retrieve_and_generate
from src.core.llm import get_llm
from src.core.router import get_router_agent, RouterAgent, reset_router_agent, set_vectorstores
from src.config.paths import (  
    RAW_DIR,
    PERSIST_DIRECTORY,
    SRC_DIR,
    CREDENTIALS_DIR
)
# Importar el router correcto (desde el atributo 'router' del módulo chat)
from src.api.routes.chat import router as chat_router

# Mover la definición de la excepción antes de usarla
class VectorstoreInitializationError(Exception):
    """Excepción personalizada para errores de inicialización del vectorstore."""
    pass

# Configuración de directorios
TEMPLATES_DIR = SRC_DIR / "api" / "templates"
STATIC_DIR = SRC_DIR / "api" / "static"

# Debug de rutas
print(f"📂 RAW_DIR: {RAW_DIR}")
print(f"📂 PERSIST_DIRECTORY: {PERSIST_DIRECTORY}")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuración de Vertex AI (simplificada) ---
def initialize_vertexai():
    """Inicializa Vertex AI de forma segura."""
    project_id = os.environ.get("GOOGLE_PROJECT_ID")
    credentials_path = str(CREDENTIALS_DIR / "proyecto-docente-453715-b625fbe2c520.json")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    if not project_id:
        try:
            import json
            with open(credentials_path, 'r') as f:
                project_id = json.load(f).get('project_id')
            if not project_id:
                raise ValueError("project_id no encontrado en credenciales.")
            os.environ["GOOGLE_PROJECT_ID"] = project_id
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
        if not os.path.exists(RAW_DIR):
            logger.error(f"No existe el directorio RAW_DIR: {RAW_DIR}")
            raise FileNotFoundError(f"Directorio RAW_DIR no encontrado: {RAW_DIR}")

        if not os.path.exists(PERSIST_DIRECTORY):
            os.makedirs(PERSIST_DIRECTORY)
            logger.info(f"Creado directorio PERSIST_DIRECTORY: {PERSIST_DIRECTORY}")

        llm = get_llm()
        if not llm:
            raise RuntimeError("No se pudo inicializar el LLM")
        print("✅ LLM inicializado")

        vectorstores = initialize_vectorstore()
        if not vectorstores:
            raise VectorstoreInitializationError(
                "No se pudo inicializar ninguna colección. Verifica data/raw"
            )

        print("\n📊 Colecciones disponibles:")
        for collection_name, vs in vectorstores.items():
            try:
                collection = vs._collection
                collection_size = collection.count()
                print(f"   ✓ {collection_name}: {collection_size} documentos")
            except Exception as e:
                print(f"   ⚠️ {collection_name}: Error al obtener tamaño - {e}")

        return llm, vectorstores, logger

    except Exception as e:
        logger.exception(f"Error fatal al inicializar el sistema: {e}")
        raise

# Inicialización de componentes
try:
    initialize_vertexai()
    llm, vectorstores, logger = initialize_system()
    
    # Configurar los vectorstores en el router
    set_vectorstores(vectorstores)
    
except Exception as e:
    logger.critical(f"Error fatal durante la inicialización: {e}")
    exit(1)

# Instanciar FastAPI
app = FastAPI()

# Montar directorio de archivos estáticos
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Configurar Jinja2Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --- Rutas de la API ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Sirve la página principal (index.html)."""
    thread_id = request.cookies.get("thread_id") or str(uuid.uuid4())[:8]
    response = templates.TemplateResponse("chat.html", {"request": request, "thread_id": thread_id})
    response.set_cookie("thread_id", thread_id)
    return response

@app.post("/consultar", response_class=JSONResponse)
async def consultar(
    pregunta: str = Form(...), 
    thread_id: str = Form(...), 
    router_agent: RouterAgent = Depends(get_router_agent)
):
    """Maneja las consultas usando el sistema RAG."""
    try:
        rag_response = retrieve_and_generate(
            vectorstores=vectorstores,
            query=pregunta,
            llm_model=llm
        )
        router_response = router_agent(pregunta, {"rag_response": rag_response})
        return {
            "respuesta": router_response,
            "fuentes": rag_response.get("sources", []),
            "documentos_relacionados": len(rag_response.get("documents", []))
        }
    except Exception as e:
        logger.exception(f"Error en /consultar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Incluir el router de chat con un prefijo claro y compartiendo las instancias
# Modificar el prefijo para evitar conflictos
app.include_router(chat_router, prefix="/api")

# Imprimir las rutas disponibles para depuración (con manejo de tipos)
print("\n--- Rutas disponibles ---")
for route in app.routes:
    if hasattr(route, "methods"):
        print(f"API Ruta: {route.path}, métodos: {route.methods}")
    elif hasattr(route, "path"):
        print(f"Mount punto: {route.path}")
    else:
        print(f"Otro tipo de ruta: {type(route)}")

@app.post("/reset")
async def reset_conversation(thread_id: str = Form(...)):
    """Reinicia la conversación."""
    if reset_router_agent(thread_id):
        logger.info(f"Conversación reiniciada para thread_id: {thread_id}")
        return {"message": "Conversación reiniciada"}
    return {"message": "No se encontró la conversación"}

@app.post("/debug-chat")
async def debug_chat(
    request: Request,
    user_message: str = Form(None),
    file: UploadFile = File(None),
    thread_id: str = Form(None)
):
    """
    Endpoint de depuración para ver qué datos se están recibiendo.
    """
    form_data = await request.form()
    return {
        "received_data": {
            "form_data": dict(form_data),
            "user_message": user_message,
            "file": file.filename if file else None,
            "thread_id": thread_id
        }
    }

# Elimina completamente estas líneas o comenta todo el bloque correctamente
# @app.post("/chat", response_class=HTMLResponse)
# async def chat_with_agent(
#     request: Request,
#     user_message: str = Form(...),
#     file: UploadFile = File(None),
#     thread_id: str = Form(...),
#     router_agent: RouterAgent = Depends(get_router_agent)
# ):
#     """
#     Chatea con el agente.
#     """
#     if file:
#         # Lógica para procesar el archivo (por ejemplo, extraer texto del PDF) se puede agregar aquí.
#         pass
# 
#     try:
#         agent_response = router_agent(user_message, {})
#         return templates.TemplateResponse(
#             "components/chat_response.html",
#             {"request": request, "agent_response": agent_response, "thread_id": thread_id},
#         )
#     except Exception as e:
#         logger.error(f"Error en la interacción con el agente: {e}")
#         raise HTTPException(status_code=500, detail=f"Error en la interacción con el agente: {e}")
