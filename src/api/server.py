import os
import uuid
import logging
from fastapi import FastAPI
from core.llm import get_llm
from core.vectorstore.loader import initialize_vectorstore
from core.agents.router_agent import create_router_agent
from core.agents.planning_agent import create_planning_agent
from core.agents.evaluation_agent import create_evaluation_agent
from core.agents.study_guide_agent import create_study_guide_agent
from api.routes import planning_router, evaluation_router, study_guide_router

app = FastAPI(
    title="RAG Docente API",
    description="API para generación de contenido educativo usando RAG",
    version="1.0.0"
)

# Configurar directorios
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PDF_DIRECTORY = os.path.join(BASE_DIR, "data", "raw", "pdf_docs")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "data", "processed", "vectorstores")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

# Inicializar componentes
llm = get_llm()
vectorstores = initialize_vectorstore(pdf_directory=PDF_DIRECTORY, persist_directory=PERSIST_DIRECTORY)
thread_id = str(uuid.uuid4())[:8]

# Crear agentes especializados a partir de los vectorstores
planning_agent = create_planning_agent(llm, vectorstores)
evaluation_agent = create_evaluation_agent(llm, vectorstores)
study_guide_agent = create_study_guide_agent(llm, vectorstores)

# Crear router principal usando la función de router que espera (llm, vectorstores, logger, thread_id)
router_agent = create_router_agent(llm, vectorstores, logger, thread_id)

# Incluir routers de rutas específicas
app.include_router(planning_router, prefix="/api/v1", tags=["planning"])
app.include_router(evaluation_router, prefix="/api/v1", tags=["evaluation"])
app.include_router(study_guide_router, prefix="/api/v1", tags=["study-guide"])

@app.post("/api/v1/generate")
async def generate_content(
    query: str,
    asignatura: str = None,
    nivel: str = None,
    mes: str = None
):
    """Endpoint principal para generar contenido educativo"""
    try:
        # Crear un estado inicial de sesión temporal para la solicitud
        session_state = {
            "pending_request": False,
            "last_query": "",
            "asignatura": asignatura,
            "nivel": nivel,
            "mes": mes,
            "tipo": None,
            "categorias": list(vectorstores.keys())
        }
        response = router_agent(query, session_state)
        return {
            "success": True,
            "needs_more_info": session_state.get("pending_request", False),
            "response": response,
            "info": session_state,
            "tipo": session_state.get("tipo", None)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado de la API"""
    return {"status": "healthy"}