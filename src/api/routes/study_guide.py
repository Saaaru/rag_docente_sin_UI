from fastapi import APIRouter, HTTPException, Depends
from src.core.agents import create_study_guide_agent
from src.api.schemas.requests import StudyGuideRequest
from src.core.llm import get_llm
from src.core.vectorstore.loader import initialize_vectorstore
from typing import Dict, Any

router = APIRouter()

async def get_study_guide_agent(
    llm = Depends(get_llm),
    vectorstores = Depends(initialize_vectorstore)
):
    return create_study_guide_agent(llm, vectorstores)

@router.post("/study-guide")
async def create_study_guide(
    request: StudyGuideRequest,
    study_guide_agent = Depends(get_study_guide_agent)
) -> Dict[str, Any]:
    """Endpoint específico para crear guías de estudio"""
    try:
        response, needs_more_info, info = study_guide_agent(
            request.query,
            request.asignatura,
            request.nivel,
            request.mes
        )
        return {"study_guide": response, "needs_more_info": needs_more_info, "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
