import logging
from typing import Dict, Any
from core.agents.router_agent import create_router_agent

# Diccionario global para almacenar los agentes por thread_id
_router_agents = {}

def get_router_agent(thread_id: str, llm, vectorstores, logger: logging.Logger):
    """Obtiene o crea un router agent para un thread_id dado."""
    global _router_agents
    if thread_id not in _router_agents:
        logger.info(f"Creando nuevo router agent para thread_id: {thread_id}")
        _router_agents[thread_id] = create_router_agent(
            llm=llm,
            vectorstores=vectorstores,
            logger=logger,
            thread_id=thread_id
        )
    return _router_agents[thread_id]

def reset_router_agent(thread_id: str) -> bool:
    """Elimina un router agent del diccionario."""
    global _router_agents
    if thread_id in _router_agents:
        del _router_agents[thread_id]
        return True
    return False