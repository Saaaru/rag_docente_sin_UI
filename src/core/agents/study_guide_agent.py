import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from src.utils.rate_limiter import rate_limited_llm_call
from src.core.vectorstore.retriever import retrieve_documents, get_context_from_documents
from typing import Dict, Optional
from langchain_chroma import Chroma
import logging

logger = logging.getLogger(__name__)

def create_study_guide_agent(llm, vectorstores: Dict[str, Chroma]):
    """
    Crea un agente especializado en guías de estudio.
    
    Args:
        llm: Modelo de lenguaje a utilizar
        vectorstores: Diccionario de vectorstores por categoría
    """
    system_prompt = """Eres un agente especializado en CREAR GUÍAS DE ESTUDIO para el sistema educativo chileno.

    DEBES verificar que tengas estos tres datos esenciales:
    1. ASIGNATURA (Lenguaje, Matemáticas, Historia, Ciencias, etc.)
    2. NIVEL (Ej: 1° básico, 7° básico, 2° medio, etc.)
    3. MES DEL AÑO ESCOLAR (opcional, pero crucial para determinar la etapa de avance)

    Si falta alguno de los datos esenciales, SOLICITA específicamente la información faltante.

    Una vez que tengas los datos, genera una guía de estudio completa con:
    - Resumen del tema principal
    - Conceptos clave (definiciones claras)
    - Ejemplos resueltos paso a paso
    - Ejercicios de práctica graduados por dificultad
    - Respuestas o soluciones a los ejercicios propuestos

    CONSIDERACIONES IMPORTANTES:
    - La guía debe reflejar la progresión natural del aprendizaje durante el año escolar. El mes indicado te ayuda a determinar en qué etapa del año estamos y, por lo tanto, qué grado de profundidad se requiere.
    - A mayor nivel educativo, se espera un contenido más avanzado y profundo, en línea con las bases curriculares.
    - La dificultad no se mide por la complejidad en sí misma, sino por la progresión de conocimientos que se deben adquirir a lo largo del año, según lo especificado en las bases curriculares.
    - Si se solicitan guías para varios meses, asegúrate de que el contenido posterior sea más avanzado que el previo.
    - La guía debe estar alineada con el currículum nacional y usar lenguaje apropiado para el nivel
    - Los ejercicios deben corresponder al avance esperado según el mes del año escolar

    Organiza la guía con títulos claros y formato amigable para estudiantes según su edad.
    """

    def study_guide_agent_executor(query: str,
                                 asignatura: Optional[str] = None,
                                 nivel: Optional[str] = None,
                                 mes: Optional[str] = None):
        """
        Ejecuta el agente de guías de estudio.
        
        Args:
            query: Consulta del usuario
            asignatura: Asignatura objetivo
            nivel: Nivel educativo
            mes: Mes del año escolar
        """
        # Verificar que tengamos acceso a los vectorstores necesarios
        required_categories = ["bases curriculares", "actividades sugeridas", "orientaciones", "propuesta"]
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
            extract_mes_prompt = [
                SystemMessage(
                    content="Extrae el mes del año escolar mencionado en esta solicitud. Si no se especifica, responde 'No especificado'."
                ),
                HumanMessage(content=query)
            ]
            mes_result = rate_limited_llm_call(llm.invoke, extract_mes_prompt)
            if "No especificado" in mes_result.content:
                meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                          "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
                mes_actual = datetime.datetime.now().month
                mes = meses[mes_actual - 1]
                print(f"\n📅 Usando mes actual para guía de estudio: {mes}")
            else:
                mes = mes_result.content.strip()

        # Si falta información, solicitarla
        if faltante:
            response = "Para crear una guía de estudio completa, necesito la siguiente información:\n\n"
            if "asignatura" in faltante:
                response += "- ¿Para qué asignatura necesitas la guía? (Lenguaje, Matemáticas, etc.)\n"
            if "nivel" in faltante:
                response += "- ¿Para qué nivel educativo? (Ej: 2° básico, 8° básico, 3° medio, etc.)\n"
            return response, True, {"asignatura": asignatura, "nivel": nivel, "mes": mes}

        try:
            # Construir query enriquecida
            enhanced_query = (
                f"guía de estudio {asignatura} {nivel} {mes} "
                f"conceptos clave ejemplos ejercicios actividades currículum"
            )

            # Definir prioridad de búsqueda por categorías
            priority_categories = ["bases curriculares", "actividades sugeridas", "orientaciones", "propuesta"]
            
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

            # Generar la guía
            guide_prompt = [
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

                Por favor, genera una guía de estudio completa que:
                1. Se adapte al nivel y asignatura específicos
                2. Incluya conceptos clave y ejemplos claros
                3. Proporcione ejercicios graduados por dificultad
                4. Se alinee con el currículum nacional
                5. Sea apropiada para el mes de {mes} en el calendario escolar
                """)
            ]

            response = rate_limited_llm_call(llm.invoke, guide_prompt)
            return response.content, False, {"asignatura": asignatura, "nivel": nivel, "mes": mes}

        except Exception as e:
            logger.exception(f"Error en study_guide_agent: {e}")
            return ("Ocurrió un error al generar la guía de estudio. Por favor, intenta nuevamente.",
                   False, None)

    return study_guide_agent_executor