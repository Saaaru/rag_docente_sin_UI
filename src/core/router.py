from typing import Dict, Any, Callable
from langchain_core.language_models import BaseChatModel
from langchain_chroma import Chroma
import logging
from src.core.llm import get_llm  # Importamos get_llm
from src.core.vectorstore.loader import initialize_vectorstore  # Importamos initialize_vectorstore
from fastapi import Depends  # Importamos Depends

logger = logging.getLogger(__name__)

# Definimos un tipo para el router agent
RouterAgent = Callable[[str, Dict[str, Any]], str]

# Usamos un diccionario para almacenar las instancias del router agent
_router_agents: Dict[str, RouterAgent] = {}

def get_router_agent(
    thread_id: str,
    llm: BaseChatModel = Depends(get_llm),  # Inyectamos llm
    vectorstores: Dict[str, Chroma] = Depends(initialize_vectorstore),  # Inyectamos vectorstores
    logger: Any = Depends(lambda: logging.getLogger(__name__)) # Inyectamos un logger
) -> RouterAgent:
    """Obtiene o crea un router agent para un thread_id dado."""
    global _router_agents

    if thread_id not in _router_agents:
        logger.info(f"Creando nuevo router agent para thread_id: {thread_id}")
        from core.agents.router_agent import create_router_agent  # Import *dentro* de la función
        _router_agents[thread_id] = create_router_agent(llm, vectorstores, logger, thread_id)
    return _router_agents[thread_id]

def reset_router_agent(thread_id: str) -> bool:
    """Reinicia el router agent para un thread_id dado."""
    global _router_agents
    if thread_id in _router_agents:
        del _router_agents[thread_id]
        logger.info(f"Router agent para thread_id {thread_id} reiniciado.")
        return True
    logger.warning(f"No se encontró router agent para thread_id {thread_id}.")
    return False
