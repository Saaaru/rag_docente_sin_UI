from fastapi import FastAPI
from core.agents import (
    create_planning_agent,
    create_evaluation_agent,
    create_study_guide_agent
)
from api.routes import planning_router, evaluation_router, study_guide_router

app = FastAPI(
    title="RAG Docente API",
    description="API para generación de contenido educativo usando RAG",
    version="1.0.0"
)

# Inicializar componentes
llm = get_llm()
embeddings = get_embeddings()
planning_agent = create_planning_agent(llm, vectorstore)
evaluation_agent = create_evaluation_agent(llm, vectorstore)
study_guide_agent = create_study_guide_agent(llm, vectorstore)
router = create_router_agent(llm, planning_agent, evaluation_agent, study_guide_agent)

# Incluir routers
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
        response, needs_info, info, tipo = router(query, asignatura, nivel, mes)
        return {
            "success": True,
            "needs_more_info": needs_info,
            "response": response,
            "info": info,
            "tipo": tipo
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
