import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from src.utils.rate_limiter import rate_limited_llm_call
from src.core.vectorstore.retriever import retrieve_documents, get_context_from_documents
from typing import Dict, Optional
from langchain_chroma import Chroma
import logging

logger = logging.getLogger(__name__)

def create_evaluation_agent(llm, vectorstores: Dict[str, Chroma]):
    """
    Crea un agente especializado en evaluaciones educativas que:
    1. Prioriza los criterios especificados por el usuario
    2. Usa configuración por defecto cuando no se especifican criterios
    3. Se adapta al nivel y asignatura
    """
    system_prompt = """Eres un experto en evaluación educativa chilena. Tu tarea es crear evaluaciones que:

    1. PRIORICEN LOS CRITERIOS DEL USUARIO:
       - Número de preguntas especificado
       - Tipos de preguntas solicitados
       - Temas o contenidos específicos
       - Formato de evaluación requerido
       - Mes del año escolar (para ajustar la dificultad según la progresión)

    2. CONFIGURACIÓN POR DEFECTO (si no se especifica):
       - 8 preguntas de selección múltiple
       - 2 preguntas de desarrollo
       - Distribución de dificultad:
         * 40% nivel básico
         * 40% nivel intermedio
         * 20% nivel avanzado

    3. ESTRUCTURA DE LA EVALUACIÓN:
       SELECCIÓN MÚLTIPLE:
       - Enunciado claro y preciso
       - 4 alternativas por pregunta
       - Solo una respuesta correcta
       - Distractores plausibles
       
       DESARROLLO:
       - Instrucciones detalladas
       - Espacio adecuado para respuesta
       - Rúbrica de evaluación
       - Puntajes asignados

    4. ELEMENTOS ADICIONALES:
       - Encabezado completo
       - Instrucciones generales
       - Tiempo sugerido
       - Puntaje total y de aprobación
       - Tabla de especificaciones

    5. CONSIDERACIONES:
       - ADAPTA LA DIFICULTAD SEGÚN EL NIVEL EDUCATIVO (1° básico = 6 años hasta 4° medio = 18 años)
       - PROGRESIÓN MENSUAL EJEMPLO: el contenido de abril debe ser más avanzado que marzo y el de mayo más avanzado que abril.
       - Alinea con los objetivos de aprendizaje según el mes del año escolar.
       - Adapta el lenguaje al nivel educativo.
       - La evaluación debe reflejar la progresión natural del aprendizaje durante el año escolar: el mes indicado te ayudará a determinar la etapa y, por ello, la profundidad de las preguntas debe ajustarse al progreso esperado según el calendario y el nivel.
       - Incluir contextos significativos, evaluar diferentes habilidades y permitir demostrar comprensión.

    IMPORTANTE: 
    - PRIORIZAR SIEMPRE los criterios específicos del usuario
    - Usar configuración por defecto solo cuando no hay especificaciones
    - Incluir retroalimentación para cada pregunta
    - Proporcionar rúbricas detalladas
    - Si la evaluación es para meses específicos, RESPETAR LA PROGRESIÓN DEL APRENDIZAJE
    """

    def evaluation_agent_executor(query: str,
                                asignatura: Optional[str] = None,
                                nivel: Optional[str] = None,
                                mes: Optional[str] = None):
        """
        Ejecuta el agente de evaluación.
        
        Args:
            query: Consulta del usuario
            asignatura: Asignatura objetivo
            nivel: Nivel educativo
            mes: Mes del año escolar
        """
        # Verificar que tengamos acceso a los vectorstores necesarios
        required_categories = ["bases curriculares", "orientaciones", "propuesta", "actividades sugeridas"]
        missing_categories = [cat for cat in required_categories if cat not in vectorstores]
        if missing_categories:
            return (f"⚠️ No se encontraron algunas categorías necesarias: {', '.join(missing_categories)}. "
                   "Por favor, verifica la disponibilidad de los documentos.", False, None)

        # Verificar información faltante
        faltante = []
        if not asignatura:
            # Intentar extraer asignatura de la consulta
            extract_prompt = [
                SystemMessage(
                    content="Extrae la asignatura mencionada en esta solicitud. Si no hay ninguna, responde 'No especificada'."),
                HumanMessage(content=query)
            ]
            asignatura_result = rate_limited_llm_call(
                llm.invoke, extract_prompt)
            if "No especificada" in asignatura_result.content:
                faltante.append("asignatura")
            else:
                asignatura = asignatura_result.content.strip()

        if not nivel:
            # Intentar extraer nivel de la consulta
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
            print(f"\n📅 Usando mes actual para evaluación: {mes}")

        # Si falta información, solicitarla
        if faltante:
            response = "Para crear una evaluación educativa completa, necesito la siguiente información:\n\n"
            if "asignatura" in faltante:
                response += "- ¿Para qué asignatura necesitas la evaluación? (Lenguaje, Matemáticas, etc.)\n"
            if "nivel" in faltante:
                response += "- ¿Para qué nivel educativo? (Ej: 2° básico, 8° básico, 3° medio, etc.)\n"
            return response, True, {"asignatura": asignatura, "nivel": nivel, "mes": mes}

        try:
            # Construir query enriquecida
            enhanced_query = (
                f"evaluación {asignatura} {nivel} {mes} "
                f"objetivos aprendizaje indicadores logro contenidos preguntas instrumento evaluación"
            )

            # Definir prioridad de búsqueda por categorías
            priority_categories = ["bases curriculares", "orientaciones", "propuesta", "actividades sugeridas"]
            
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

            # Generar la evaluación
            evaluation_prompt = [
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

                Por favor, genera una evaluación que:
                1. Se adapte al nivel y asignatura específicos
                2. Use la configuración por defecto para aspectos no especificados
                3. Se alinee con el currículum nacional
                4. Incluya instrucciones claras y rúbricas de evaluación
                5. Sea apropiada para el mes de {mes} en el calendario escolar
                """)
            ]

            response = rate_limited_llm_call(llm.invoke, evaluation_prompt)
            return response.content, False, {"asignatura": asignatura, "nivel": nivel, "mes": mes}

        except Exception as e:
            logger.exception(f"Error en evaluation_agent: {e}")
            return ("Ocurrió un error al generar la evaluación. Por favor, intenta nuevamente.",
                   False, None)

    return evaluation_agent_executor