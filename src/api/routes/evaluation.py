from fastapi import APIRouter, HTTPException, Depends
from src.core.agents import create_evaluation_agent
from src.api.schemas.requests import EvaluationRequest
from src.core.llm import get_llm
from src.core.vectorstore.loader import initialize_vectorstore
from typing import Dict, Any

router = APIRouter()

async def get_evaluation_agent(
    llm = Depends(get_llm),
    vectorstores = Depends(initialize_vectorstore)
):
    return create_evaluation_agent(llm, vectorstores)

@router.post("/evaluation")
async def create_evaluation(
    request: EvaluationRequest,
    evaluation_agent = Depends(get_evaluation_agent)
) -> Dict[str, Any]:
    """Endpoint específico para crear evaluaciones"""
    try:
        response, needs_more_info, info = evaluation_agent(
            request.query,
            request.asignatura,
            request.nivel,
            request.mes
        )
        return {"evaluation": response, "needs_more_info": needs_more_info, "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
