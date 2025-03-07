import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from utils.rate_limiter import rate_limited_llm_call
from core.vectorstore.retriever import create_enhanced_retriever_tool

def create_planning_agent(llm, vectorstore):
    """
    Crea un agente especializado en planificaciones educativas que considera:
    1. El contexto mensual chileno (fechas importantes, eventos, clima, etc.)
    2. Progresión ascendente del aprendizaje durante el año
    3. Adaptación al nivel y asignatura específicos
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
       - Recursos necesarios, obten la información de los libros de texto.
       - Adecuaciones según contexto mensual

    4. CONSIDERACIONES:
       - Alinear con bases curriculares vigentes
       - Adaptar al nivel y asignatura
       - Incluir habilidades transversales
       - Considerar diversidad de estudiantes
       - Incorporar elementos culturales chilenos

    IMPORTANTE: 
    - ADAPTA LA DIFICULTAD SEGÚN EL NIVEL EDUCATIVO (1° básico = 6 años hasta 4° medio = 18 años)
    - Asegura que cada mes construya sobre el anterior
    - Si te solicitan planificación para varios meses (ej: marzo-abril), GARANTIZA que el segundo mes 
      tenga mayor complejidad y se base en lo aprendido en el mes anterior
    - Incluye actividades contextualizadas al mes
    - Considera el clima, estación del año, eventos relevantes del calendario escolar.
    """

    def planning_agent_executor(query, asignatura=None, nivel=None, mes=None):
        # Verificar información faltante
        faltante = []
        if not asignatura:
            extract_prompt = [
                SystemMessage(
                    content="Extrae la asignatura mencionada en esta solicitud. Si no hay ninguna, solicita nuevamente la información completa."),
                HumanMessage(content=query)
            ]
            asignatura_result = rate_limited_llm_call(
                llm.invoke, extract_prompt)
            if "No especificada" in asignatura_result.content:
                faltante.append("asignatura")
            else:
                asignatura = asignatura_result.content.strip()

        if not nivel:
            extract_prompt = [
                SystemMessage(
                    content="Extrae el nivel educativo mencionado en esta solicitud (ej: 5° básico, 2° medio). Si no hay ninguna, solicita nuevamente la información completa."),
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

        # Si tenemos toda la información, generar la planificación
        enhanced_query = f"Crear planificación para la asignatura de {asignatura} para el nivel {nivel} para el mes de {mes}"
        
        # Buscar información relevante en el vectorstore
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10}
        )

        retrieved_docs = retriever.invoke(enhanced_query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Generar la planificación - Eliminamos variables no definidas
        planning_prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            SOLICITUD: {query}
            ASIGNATURA: {asignatura}
            NIVEL: {nivel}
            MES: {mes}

            CONTEXTO CURRICULAR:
            {context}

            Por favor, genera una planificación educativa completa considerando:
            1. El nivel y asignatura específicos
            2. El contexto curricular proporcionado
            3. La progresión del aprendizaje
            4. El contexto educativo chileno
            5. El mes específico: {mes}
            """)
        ]

        response = rate_limited_llm_call(llm.invoke, planning_prompt)
        return response.content, False, {"asignatura": asignatura, "nivel": nivel, "mes": mes}

    return planning_agent_executor