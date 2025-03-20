from fastapi import APIRouter, HTTPException, Depends
from src.core.agents import create_planning_agent
from src.api.schemas.requests import PlanningRequest
from src.core.llm import get_llm
from src.core.vectorstore.loader import initialize_vectorstore
from typing import Optional, Tuple, Dict, Any

router = APIRouter()

# Función para obtener el agente de planificación (con inyección de dependencias)
async def get_planning_agent(
    llm=Depends(get_llm),
    vectorstores=Depends(initialize_vectorstore)
):
    # Aquí podrías agregar lógica para manejar diferentes instancias de agentes si fuera necesario
    return create_planning_agent(llm, vectorstores)

@router.post("/planning")
async def create_planning(
    request: PlanningRequest,
    planning_agent=Depends(get_planning_agent)  # Inyectamos el agente
) -> Dict[str, Any]:
    """Endpoint específico para crear planificaciones"""
    try:
        # Usar el agente de planificación (obtenido por inyección)
        response, needs_more_info, info = planning_agent(
            request.query,
            request.asignatura,
            request.nivel,
            request.mes
        )
        return {"planning": response, "needs_more_info": needs_more_info, "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
