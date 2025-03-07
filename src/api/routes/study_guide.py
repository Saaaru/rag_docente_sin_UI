from fastapi import APIRouter, HTTPException
from core.agents import create_study_guide_agent
from api.schemas.requests import StudyGuideRequest

router = APIRouter()

@router.post("/study-guide")
async def create_study_guide(request: StudyGuideRequest):
    """Endpoint específico para crear guías de estudio"""
    try:
        response, _, info = create_study_guide_agent(
            request.query,
            request.asignatura,
            request.nivel,
            request.mes
        )
        return {"study_guide": response, "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
