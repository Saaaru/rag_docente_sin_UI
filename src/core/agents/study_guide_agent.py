import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from utils.rate_limiter import rate_limited_llm_call
from core.vectorstore.retriever import retrieve_documents, get_context_from_documents
from typing import Dict, Optional
from langchain_chroma import Chroma

def create_study_guide_agent(llm, vectorstores: Dict[str, Chroma]):
    """
    Crea un agente especializado en guías de estudio.
    
    Args:
        llm: Modelo de lenguaje a utilizar
        vectorstores: Diccionario de vectorstores por categoría
    """
    system_prompt = """Eres un agente especializado en CREAR GUÍAS DE ESTUDIO para el sistema educativo chileno.

    DEBES verificar que tengas estos dos datos esenciales:
    1. ASIGNATURA (Lenguaje, Matemáticas, Historia, Ciencias, etc.)
    2. NIVEL (Específico: 1° básico, 7° básico, 2° medio, etc.)
    3. MES DEL AÑO ESCOLAR (opcional, pero importante para la progresión)

    Si falta alguno de los datos esenciales, SOLICITA específicamente la información faltante.

    Una vez que tengas los datos, genera una guía de estudio completa con:
    - Resumen del tema principal
    - Conceptos clave (definiciones claras)
    - Ejemplos resueltos paso a paso
    - Ejercicios de práctica graduados por dificultad
    - Respuestas o soluciones a los ejercicios propuestos

    CONSIDERACIONES IMPORTANTES:
    - ADAPTA LA DIFICULTAD SEGÚN EL NIVEL EDUCATIVO (1° básico = 6 años hasta 4° medio = 18 años)
    - PROGRESIÓN MENSUAL EJEMPLO: el contenido de abril debe ser más avanzado que marzo y el de mayo más avanzado que abril.
    - Si te solicitan guías para varios meses, ASEGURA que el material posterior sea más complejo
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
            meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", 
                    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
            mes_actual = datetime.datetime.now().month
            mes = meses[mes_actual - 1]
            print(f"\n📅 Usando mes actual para guía de estudio: {mes}")

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
            print(f"❌ Error en study_guide_agent: {e}")
            return ("Ocurrió un error al generar la guía de estudio. Por favor, intenta nuevamente.",
                   False, None)

    return study_guide_agent_executor