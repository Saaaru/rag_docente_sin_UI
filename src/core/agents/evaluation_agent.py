import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from utils.rate_limiter import rate_limited_llm_call
from core.vectorstore.retriever import create_enhanced_retriever_tool

def create_evaluation_agent(llm, vectorstore):
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
       - Mes del año escolar (para ajustar dificultad según progresión)

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
       - Alinear con objetivos de aprendizaje según mes del año escolar
       - Adaptar lenguaje al nivel
       - Incluir contextos significativos
       - Evaluar diferentes habilidades
       - Permitir demostrar comprensión

    IMPORTANTE: 
    - PRIORIZAR SIEMPRE los criterios específicos del usuario
    - Usar configuración por defecto solo cuando no hay especificaciones
    - Incluir retroalimentación para cada pregunta
    - Proporcionar rúbricas detalladas
    - Si la evaluación es para meses específicos, RESPETAR LA PROGRESIÓN DEL APRENDIZAJE
    """

    def evaluation_agent_executor(query, asignatura=None, nivel=None, mes=None):
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

        # Si tenemos toda la información, generar la evaluación
        enhanced_query = f"Crear evaluación para la asignatura de {asignatura} para el nivel {nivel} para el mes de {mes}"

        # Buscar información relevante en el vectorstore
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10}
        )

        retrieved_docs = retriever.invoke(enhanced_query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Generar la evaluación - Eliminamos variables no definidas
        evaluation_prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            SOLICITUD: {query}
            ASIGNATURA: {asignatura}
            NIVEL: {nivel}
            MES: {mes}

            CONTEXTO CURRICULAR:
            {context}

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

    return evaluation_agent_executor