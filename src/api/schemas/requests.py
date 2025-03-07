from pydantic import BaseModel
from typing import Optional

class BaseRequest(BaseModel):
    """Modelo base para todas las solicitudes"""
    query: str
    asignatura: Optional[str] = None
    nivel: Optional[str] = None
    mes: Optional[str] = None

class PlanningRequest(BaseRequest):
    """Solicitud para planificaciones"""
    pass

class EvaluationRequest(BaseRequest):
    """Solicitud para evaluaciones"""
    pass

class StudyGuideRequest(BaseRequest):
    """Solicitud para guías de estudio"""
    pass
