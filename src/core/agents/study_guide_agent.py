import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from utils.rate_limiter import rate_limited_llm_call
from core.vectorstore.retriever import retrieve_documents, get_context_from_documents

def create_study_guide_agent(llm, vectorstore):
    """
    Crea un agente especializado en guías de estudio.
    Verifica que tenga asignatura y nivel antes de generar contenido.
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

    def study_guide_agent_executor(query, asignatura=None, nivel=None, mes=None):
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

        # Si tenemos toda la información, generar la guía
        enhanced_query = f"guía de estudio {asignatura} {nivel} {mes} conceptos clave ejemplos ejercicios actividades currículum"

        # Recopilar información de manera estratégica
        # Priorizar bases curriculares, actividades y orientaciones
        priority_categories = ["bases curriculares", "actividades sugeridas", "orientaciones", "propuesta"]
        
        # Utilizar retrieve_documents para obtener documentos relevantes
        retrieved_docs = retrieve_documents(
            vectorstore=vectorstore,
            query=enhanced_query,
            categories=priority_categories,
            k=7
        )
        
        # Extraer contexto de los documentos recuperados
        context, sources = get_context_from_documents(retrieved_docs)
        
        source_text = ", ".join(sources) if sources else "documentos curriculares disponibles"

        # Generar la guía
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Solicitud: {query}
            Asignatura: {asignatura}
            Nivel: {nivel}
            Mes: {mes}

            Contenidos y objetivos de aprendizaje relevantes:
            {context}
            
            Fuentes de información consultadas:
            {source_text}
            
            Genera una guía de estudio completa basada en el currículum nacional chileno, 
            adaptada al nivel educativo solicitado y que cubra los contenidos correspondientes
            al mes indicado según la progresión del aprendizaje.
            """)
        ]

        response = rate_limited_llm_call(llm.invoke, prompt)
        return response.content, False, {"asignatura": asignatura, "nivel": nivel, "mes": mes}

    return study_guide_agent_executor