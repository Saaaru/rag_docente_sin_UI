from fastapi import APIRouter, HTTPException
from core.agents import create_evaluation_agent
from api.schemas.requests import EvaluationRequest

router = APIRouter()

@router.post("/evaluation")
async def create_evaluation(request: EvaluationRequest):
    """Endpoint específico para crear evaluaciones"""
    try:
        response, _, info = create_evaluation_agent(
            request.query,
            request.asignatura,
            request.nivel,
            request.mes
        )
        return {"evaluation": response, "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
