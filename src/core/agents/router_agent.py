import datetime
from typing import Dict, Optional, List, Any
from src.utils.conversation import format_and_save_conversation
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from src.utils.rate_limiter import rate_limited_llm_call
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_chroma import Chroma
import json
import logging
from core.llm import get_llm

# Importar las funciones de creación de los agentes especializados
# YA NO ES NECESARIO IMPORTARLOS AQUÍ
# from core.agents.planning_agent import create_planning_agent
# from core.agents.evaluation_agent import create_evaluation_agent
# from core.agents.study_guide_agent import create_study_guide_agent

_agent_cache = {}  # Diccionario para almacenar las instancias de los agentes

class RouterChatModel(BaseChatModel):
    """Modelo de chat personalizado para el router que interpreta las consultas de usuario."""
    
    def __init__(self, llm, logger):
        super().__init__()
        self._llm = llm
        self._logger = logger
    
    @property
    def llm(self):
        return self._llm
        
    @property
    def logger(self):
        return self._logger

    @property
    def _llm_type(self) -> str:
        return "router-chat-model"

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        """Genera una interpretación estructurada de la consulta del usuario."""
        
        system_prompt = """Eres un agente router inteligente especializado en educación.
        Tu tarea es interpretar consultas de usuarios y determinar:

        1. Si la consulta requiere uno de nuestros agentes especializados:
           - PLANIFICACION: Para crear planificaciones educativas
           - EVALUACION: Para crear evaluaciones
           - GUIA: Para crear guías de estudio
           
        2. Si es una consulta que requiere agente especializado, identifica:
           - Asignatura mencionada (Lenguaje, Matemáticas, Historia, Ciencias)
           - Nivel educativo (1° a 6° Básico, 7° básico a 2° medio, 3° a 4° medio)
           - Si es una respuesta a una pregunta previa
        
        3. Si NO requiere agente especializado:
           - Determina si es una consulta general
           - Prepara una respuesta orientativa

        IMPORTANTE: DEBES responder SIEMPRE en formato JSON con esta estructura exacta:
        {
            "requiere_agente": false,
            "tipo_contenido": "NINGUNO",
            "asignatura": null,
            "nivel": null,
            "es_respuesta": false,
            "respuesta_directa": "Mensaje apropiado para el usuario"
        }

        Para saludos o consultas generales, usa respuesta_directa con un mensaje amigable.
        """

        try:
            # Asegurar que usamos el último mensaje
            last_message = messages[-1].content if messages else ""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=last_message)
            ]

            response = self.llm.invoke(messages)
            content = response.content if isinstance(response, AIMessage) else str(response)
            
            # Intentar extraer JSON de la respuesta
            try:
                # Buscar el primer '{' y el último '}'
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = content[start:end]
                    interpretation = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except:
                # Si falla, crear respuesta por defecto
                interpretation = {
                    "requiere_agente": False,
                    "tipo_contenido": "NINGUNO",
                    "asignatura": None,
                    "nivel": None,
                    "es_respuesta": False,
                    "respuesta_directa": "¡Hola! Soy tu asistente educativo. ¿En qué puedo ayudarte? Puedo crear planificaciones, evaluaciones o guías de estudio."
                }

            return ChatResult(generations=[
                ChatGeneration(message=AIMessage(content=json.dumps(interpretation)))
            ])

        except Exception as e:
            self.logger.error(f"Error en RouterChatModel: {str(e)}")
            # Retornar respuesta por defecto en caso de error
            default_response = {
                "requiere_agente": False,
                "tipo_contenido": "NINGUNO",
                "asignatura": None,
                "nivel": None,
                "es_respuesta": False,
                "respuesta_directa": "¡Hola! ¿Cómo puedo ayudarte? Puedo crear planificaciones, evaluaciones o guías de estudio."
            }
            return ChatResult(generations=[
                ChatGeneration(message=AIMessage(content=json.dumps(default_response)))
            ])