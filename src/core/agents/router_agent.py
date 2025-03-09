import datetime
from typing import Optional
from utils.conversation import format_and_save_conversation
from langchain_core.messages import SystemMessage, HumanMessage
from utils.rate_limiter import rate_limited_llm_call
import json, re
from typing import List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

# Importar las *funciones de creación* de los agentes especializados
from core.agents.planning_agent import create_planning_agent
from core.agents.evaluation_agent import create_evaluation_agent
from core.agents.study_guide_agent import create_study_guide_agent

class RouterChatModel(BaseChatModel):
    """Modelo de chat personalizado para el router que interpreta las consultas de usuario."""
    
    def __init__(self, llm, logger):
        super().__init__()
        self.llm = llm
        self.logger = logger
        
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
        
        Responde en formato JSON estricto:
        {
            "requiere_agente": boolean,
            "tipo_contenido": "PLANIFICACION|EVALUACION|GUIA|NINGUNO",
            "asignatura": string | null,
            "nivel": string | null,
            "es_respuesta": boolean,
            "respuesta_directa": string | null
        }"""

        messages = [
            SystemMessage(content=system_prompt),
            messages[-1]  # Solo usamos el último mensaje para la interpretación
        ]

        response = self.llm.invoke(messages)
        
        try:
            # Convertir la respuesta a JSON
            content = response.content if isinstance(response, AIMessage) else str(response)
            interpretation = json.loads(content)
            
            return ChatResult(generations=[
                ChatGeneration(message=AIMessage(content=json.dumps(interpretation)))
            ])
        except Exception as e:
            self.logger.error(f"Error interpretando respuesta del LLM: {e}")
            return ChatResult(generations=[
                ChatGeneration(message=AIMessage(content=json.dumps({
                    "requiere_agente": False,
                    "tipo_contenido": "NINGUNO",
                    "asignatura": None,
                    "nivel": None,
                    "es_respuesta": False,
                    "respuesta_directa": "No pude entender tu consulta. ¿Podrías reformularla?"
                })))
            ])

def create_router_agent(llm, planning_agent, evaluation_agent, study_guide_agent, logger, thread_id):
    """Crea un agente router que utiliza el LLM para interpretar consultas."""
    
    router_chat = RouterChatModel(llm=llm, logger=logger)
    
    def router(user_input: str, session_state: dict) -> str:
        """Función principal del router."""
        logger.info(f"Procesando consulta: {user_input}")

        try:
            # Obtener interpretación usando el modelo de chat
            result = router_chat.invoke([HumanMessage(content=user_input)])
            interpretation = json.loads(result.content if isinstance(result, AIMessage) else result)
            
            logger.debug(f"Interpretación: {interpretation}")
            
            # Si no requiere agente especializado, dar respuesta directa
            if not interpretation["requiere_agente"]:
                return interpretation["respuesta_directa"]

            # Si es una nueva consulta (no es respuesta), actualizar estado
            if not interpretation["es_respuesta"]:
                session_state.clear()
                session_state.update({
                    "pending_request": False,
                    "last_query": user_input,
                    "asignatura": interpretation["asignatura"],
                    "nivel": interpretation["nivel"],
                    "mes": None,
                    "tipo": interpretation["tipo_contenido"]
                })

            # Actualizar información del estado si se proporciona
            if interpretation["asignatura"]:
                session_state["asignatura"] = interpretation["asignatura"]
            if interpretation["nivel"]:
                session_state["nivel"] = interpretation["nivel"]

            # Verificar si necesitamos más información
            if not session_state.get("asignatura"):
                session_state["pending_request"] = True
                return "¿Para qué asignatura necesitas este material?"
            if not session_state.get("nivel"):
                session_state["pending_request"] = True
                return "¿Para qué nivel educativo necesitas este material?"

            # Asegurar que tenemos el mes
            if not session_state.get("mes"):
                current_month = datetime.datetime.now().month
                months = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                         "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
                session_state["mes"] = months[current_month - 1]

            # Seleccionar y llamar al agente apropiado
            agent_to_use = {
                "PLANIFICACION": planning_agent,
                "EVALUACION": evaluation_agent,
                "GUIA": study_guide_agent
            }.get(interpretation["tipo_contenido"], study_guide_agent)

            response = agent_to_use(
                user_input,
                session_state["asignatura"],
                session_state["nivel"],
                session_state["mes"]
            )

            # Registrar la conversación
            format_and_save_conversation(
                f"{user_input} (Asignatura: {session_state['asignatura']}, "
                f"Nivel: {session_state['nivel']}, Mes: {session_state['mes']})",
                response,
                thread_id
            )

            # Limpiar el estado después de una respuesta exitosa
            session_state.clear()
            session_state.update({
                "pending_request": False,
                "last_query": "",
                "asignatura": None,
                "nivel": None,
                "mes": None,
                "tipo": None
            })

            return response

        except Exception as e:
            logger.error(f"Error en router: {e}")
            return "Lo siento, hubo un problema al procesar tu solicitud. ¿Podrías reformularla?"

    return router