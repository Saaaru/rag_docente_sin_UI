import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from src.utils.rate_limiter import rate_limited_llm_call
from src.core.vectorstore.retriever import retrieve_documents, get_context_from_documents
from typing import Dict, Optional
from langchain_chroma import Chroma
import logging

logger = logging.getLogger(__name__)

class PlanningAgentError(Exception):
    """Excepción personalizada para errores en el agente de planificación."""
    pass

def create_planning_agent(llm, vectorstores: Dict[str, Chroma]):
    """
    Crea un agente especializado en planificaciones educativas.
    
    Args:
        llm: Modelo de lenguaje a utilizar
        vectorstores: Diccionario de vectorstores por categoría
    """
    system_prompt = """Eres un experto en planificación educativa chilena. Tu tarea es crear planificaciones que:

    1. CONSIDEREN EL CONTEXTO CHILENO MENSUAL:
       MARZO: Inicio año escolar, adaptación, diagnóstico
       ABRIL: Fiestas patrias otoñales
       MAYO: Día del estudiante, Glorias Navales
       JUNIO: Pueblos originarios, Día del padre, inicio invierno
       JULIO: Vacaciones de invierno, evaluación primer semestre
       AGOSTO: Mes de la solidaridad, retorno a clases
       SEPTIEMBRE: Fiestas patrias, folclore
       OCTUBRE: Encuentro de dos mundos, primavera
       NOVIEMBRE: Preparación cierre año escolar
       DICIEMBRE: Evaluaciones finales, actividades cierre

    2. ASEGUREN PROGRESIÓN DEL APRENDIZAJE:
       - PROGRESIÓN POR NIVEL: desde 1° básico (6 años) hasta 4° medio (18 años)
       - PROGRESIÓN MENSUAL EJEMPLO: el contenido de abril debe ser más avanzado que marzo y el de mayo más avanzado que abril.
       - Partir de conocimientos básicos/diagnóstico
       - Avanzar gradualmente en complejidad
       - Conectar contenidos entre meses
       - Reforzar aprendizajes previos
       - Introducir nuevos desafíos progresivamente

    3. ESTRUCTURA DE LA PLANIFICACIÓN:
       - Objetivo general (del currículum nacional)
       - Objetivos específicos (entre 3 a 5)
       - Contenidos y habilidades (entre 3 a 5)
       - Actividades sugeridas (mínimo 3)
       - Evaluación formativa
       - Recursos necesarios
       - Adecuaciones según contexto mensual

    4. CONSIDERACIONES:
       - Alinear con bases curriculares vigentes
       - Adaptar al nivel y asignatura
       - Incluir habilidades transversales
       - Considerar diversidad de estudiantes
       - Incorporar elementos culturales chilenos
       - La planificación debe comprender que el avance del aprendizaje es progresivo: el contenido y las actividades deben incrementarse en complejidad y profundidad conforme se avanza en el año escolar y con el nivel educativo.

    IMPORTANTE: 
    - ADAPTA LA DIFICULTAD SEGÚN EL NIVEL EDUCATIVO (1° básico = 6 años hasta 4° medio = 18 años)
    - Asegura que cada mes construya sobre el anterior
    - Si te solicitan planificación para varios meses (ej: marzo-abril), GARANTIZA que el segundo mes 
      tenga mayor complejidad y se base en lo aprendido en el mes anterior
    - Incluye actividades contextualizadas al mes
    - Considera el clima, estación del año, eventos relevantes del calendario escolar.
    """

    def planning_agent_executor(query: str, 
                              asignatura: Optional[str] = None, 
                              nivel: Optional[str] = None, 
                              mes: Optional[str] = None):
        """
        Ejecuta el agente de planificación.
        
        Args:
            query: Consulta del usuario
            asignatura: Asignatura objetivo
            nivel: Nivel educativo
            mes: Mes del año escolar
        """
        try:
            # Verificar que tengamos acceso a los vectorstores necesarios
            required_categories = ["bases curriculares", "orientaciones", "propuesta"]
            missing_categories = [cat for cat in required_categories if cat not in vectorstores]
            if missing_categories:
                return (f"⚠️ No se encontraron algunas categorías necesarias: {', '.join(missing_categories)}. "
                       "Por favor, verifica la disponibilidad de los documentos.", False, None)

            # Verificar información faltante
            faltante = []
            if not asignatura:
                extract_prompt = [
                    SystemMessage(
                        content="Extrae la asignatura mencionada en esta solicitud. Si no hay ninguna, responde 'No especificada'."),
                    HumanMessage(content=query)
                ]
                asignatura_result = rate_limited_llm_call(llm.invoke, extract_prompt)
                if "No especificada" in asignatura_result.content:
                    faltante.append("asignatura")
                else:
                    asignatura = asignatura_result.content.strip()

            if not nivel:
                extract_prompt = [
                    SystemMessage(
                        content="Extrae el nivel educativo mencionado en esta solicitud (ej: 5° básico, 2° medio). Si no hay ninguno, responde 'No especificado'."),
                    HumanMessage(content=query)
                ]
                nivel_result = rate_limited_llm_call(llm.invoke, extract_prompt)
                if "No especificado" in nivel_result.content:
                    faltante.append("nivel")
                else:
                    nivel = nivel_result.content.strip()
            
            # Si no se ha especificado un mes, usamos el mes actual
            if not mes:
                meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", 
                        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
                mes_actual = datetime.datetime.now().month
                mes = meses[mes_actual - 1]
                print(f"\n📅 Usando mes actual para planificación: {mes}")

            # Si falta información, solicitarla
            if faltante:
                response = "Para crear una planificación educativa completa, necesito la siguiente información:\n\n"
                if "asignatura" in faltante:
                    response += "- ¿Para qué asignatura necesitas la planificación? (Lenguaje, Matemáticas, etc.)\n"
                if "nivel" in faltante:
                    response += "- ¿Para qué nivel educativo? (Ej: 2° básico, 8° básico, 3° medio, etc.)\n"
                return response, True, {"asignatura": asignatura, "nivel": nivel, "mes": mes}

            # Construir query enriquecida
            enhanced_query = (
                f"planificación curricular {asignatura} {nivel} {mes} "
                f"objetivos aprendizaje contenidos habilidades actividades evaluación"
            )

            # Definir prioridad de búsqueda por categorías
            priority_categories = ["bases curriculares", "orientaciones", "propuesta"]
            
            print(f"\n🔍 Buscando información relevante en: {', '.join(priority_categories)}")
            
            # Recuperar documentos relevantes
            retrieved_docs = retrieve_documents(
                vectorstores=vectorstores,
                query=enhanced_query,
                categories=priority_categories,
                k=7
            )

            if not retrieved_docs:
                return ("No se encontró información suficiente en los documentos curriculares. "
                       "Por favor, verifica los criterios de búsqueda.", False, None)

            # Extraer contexto y fuentes
            context, sources = get_context_from_documents(retrieved_docs)
            source_text = ", ".join(sources) if sources else "documentos curriculares disponibles"

            print(f"📚 Fuentes consultadas: {source_text}")

            # Generar la planificación
            planning_prompt = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
                SOLICITUD: {query}
                ASIGNATURA: {asignatura}
                NIVEL: {nivel}
                MES: {mes}

                CONTEXTO CURRICULAR:
                {context}

                FUENTES CONSULTADAS:
                {source_text}

                Por favor, genera una planificación educativa completa considerando:
                1. El nivel y asignatura específicos
                2. El contexto curricular proporcionado
                3. La progresión del aprendizaje a lo largo del año escolar
                4. El contexto educativo chileno
                5. El mes específico: {mes}
                """)
            ]

            try:
                response = rate_limited_llm_call(llm.invoke, planning_prompt)
                return response.content, False, {"asignatura": asignatura, "nivel": nivel, "mes": mes}
            except Exception as e:
                logger.error(f"Error al generar planificación: {e}")
                raise PlanningAgentError("Error al generar la planificación.") from e

        except PlanningAgentError as e:
            logger.error(f"Error en planning_agent: {e}")
            return str(e), False, None  # Retornar el mensaje de error
        except Exception as e:
            logger.exception(f"Error inesperado en planning_agent: {e}")
            return "Error inesperado al generar la planificación.", False, None

    return planning_agent_executor