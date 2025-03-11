import datetime
from typing import Dict, Optional, List, Any
from utils.conversation import format_and_save_conversation
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from utils.rate_limiter import rate_limited_llm_call
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_chroma import Chroma
import json
import logging

# Importar las funciones de creación de los agentes especializados
from core.agents.planning_agent import create_planning_agent
from core.agents.evaluation_agent import create_evaluation_agent
from core.agents.study_guide_agent import create_study_guide_agent

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

def create_router_agent(llm, vectorstores: Dict[str, Chroma], logger: logging.Logger, thread_id: str):
    """
    Crea un agente router que coordina los agentes especializados.
    
    Args:
        llm: Modelo de lenguaje a utilizar
        vectorstores: Diccionario de vectorstores por categoría
        logger: Logger para registro de eventos
        thread_id: Identificador único de la conversación
    """
    router_chat = RouterChatModel(llm=llm, logger=logger)
    
    # Crear instancias de los agentes especializados
    planning_agent = create_planning_agent(llm, vectorstores)
    evaluation_agent = create_evaluation_agent(llm, vectorstores)
    study_guide_agent = create_study_guide_agent(llm, vectorstores)
    
    def router(user_input: str, session_state: Dict[str, Any]) -> str:
        """
        Función principal del router.
        
        Args:
            user_input: Consulta del usuario
            session_state: Estado actual de la sesión
        """
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
                    "tipo": interpretation["tipo_contenido"],
                    "categorias": list(vectorstores.keys())  # Añadir categorías disponibles
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
            agent_map = {
                "PLANIFICACION": planning_agent,
                "EVALUACION": evaluation_agent,
                "GUIA": study_guide_agent
            }
            
            selected_agent = agent_map.get(interpretation["tipo_contenido"])
            if not selected_agent:
                logger.error(f"Tipo de contenido no válido: {interpretation['tipo_contenido']}")
                return "Lo siento, no puedo procesar este tipo de solicitud."

            logger.info(f"Ejecutando agente: {interpretation['tipo_contenido']}")
            
            response, needs_more_info, updated_state = selected_agent(
                user_input,
                session_state["asignatura"],
                session_state["nivel"],
                session_state["mes"]
            )

            # Si el agente necesita más información
            if needs_more_info:
                session_state["pending_request"] = True
                return response

            # Registrar la conversación exitosa
            format_and_save_conversation(
                f"{user_input} (Asignatura: {session_state['asignatura']}, "
                f"Nivel: {session_state['nivel']}, Mes: {session_state['mes']})",
                response,
                thread_id
            )

            # Actualizar el estado con la información del agente
            if updated_state:
                session_state.update(updated_state)
                session_state["pending_request"] = False
                session_state["last_query"] = ""

            return response

        except Exception as e:
            logger.error(f"Error en router: {e}")
            return "Lo siento, hubo un problema al procesar tu solicitud. ¿Podrías reformularla?"

    return router