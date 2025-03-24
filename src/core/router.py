from typing import Dict, Any, Callable
from langchain_core.language_models import BaseChatModel
from langchain_chroma import Chroma
import logging
from src.core.llm import get_llm
# Remover la importación de initialize_vectorstore para evitar dependencia circular
# from src.core.vectorstore.loader import initialize_vectorstore
from fastapi import Depends, Form

logger = logging.getLogger(__name__)

# Definimos un tipo para el router agent
RouterAgent = Callable[[str, Dict[str, Any]], str]

# Diccionario para almacenar instancias de router agents
_router_agents: Dict[str, RouterAgent] = {}

# Variable global para almacenar los vectorstores
_vectorstores = None

# Función para configurar los vectorstores (llamada desde main.py)
def set_vectorstores(vectorstores):
    global _vectorstores
    _vectorstores = vectorstores

def get_router_agent(
    thread_id: str = Form(...),
    llm: BaseChatModel = Depends(get_llm)
) -> RouterAgent:
    """Obtiene o crea un router agent para un thread_id dado."""
    global _router_agents, _vectorstores

    if thread_id not in _router_agents:
        logger.info(f"Creando nuevo router agent para thread_id: {thread_id}")
        # Importación local para evitar dependencias circulares
        from src.core.agents.router_agent import create_router_agent
        
        # Verificar que los vectorstores estén disponibles
        if _vectorstores is None:
            logger.error("Error: Vectorstores no inicializados")
            # Usar un valor vacío como fallback
            _vectorstores = {}
            
        _router_agents[thread_id] = create_router_agent(llm, _vectorstores, logger, thread_id)
    
    return _router_agents[thread_id]

def reset_router_agent(thread_id: str) -> bool:
    """Reinicia un router agent para un thread_id dado."""
    global _router_agents
    
    if thread_id in _router_agents:
        logger.info(f"Eliminando router agent para thread_id: {thread_id}")
        del _router_agents[thread_id]
        return True
    return False
