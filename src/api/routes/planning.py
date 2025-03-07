from fastapi import APIRouter, HTTPException
from core.agents import create_planning_agent
from api.schemas.requests import PlanningRequest

router = APIRouter()

@router.post("/planning")
async def create_planning(request: PlanningRequest):
    """Endpoint específico para crear planificaciones"""
    try:
        # Usar el agente de planificación
        response, _, info = create_planning_agent(
            request.query,
            request.asignatura,
            request.nivel,
            request.mes
        )
        return {"planning": response, "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
