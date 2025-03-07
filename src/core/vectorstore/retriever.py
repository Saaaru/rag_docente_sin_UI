from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils.rate_limiter import rate_limited_llm_call


def create_filtered_retriever(vectorstore, filter_text, k=2):
    """
    Retriever mejorado con filtrado más preciso
    """
    def filtered_search(query):
        # Usar MMR para diversidad en resultados
        docs = vectorstore.max_marginal_relevance_search(
            query=query,
            k=k * 2,
            fetch_k=k * 4,
            lambda_mult=0.7,
            filter={"source": {"$contains": filter_text.lower()}}
        )
        
        # Aplicar filtrado adicional si es necesario
        filtered_docs = [
            doc for doc in docs 
            if hasattr(doc, 'metadata') 
            and 'source' in doc.metadata
        ]
        
        return filtered_docs[:k]
    
    return filtered_search

def direct_answer_generator(llm, query, documents, source_documents=None, conversation_history=None):
    """
    Genera una respuesta directa basada en los documentos recuperados y el historial de conversación.
    Verifica que la solicitud incluya asignatura y nivel antes de generar contenido.
    """
    # Interpretar el tipo de contenido y extraer detalles
    interpret_prompt = [
        SystemMessage(content="""Analiza la consulta del usuario e identifica:
        1. Tipo de contenido solicitado (PLANIFICACIÓN, EVALUACIÓN o GUÍA)
        2. Asignatura mencionada
        3. Nivel educativo mencionado

        NIVELES VÁLIDOS:
        - Sala Cuna (0-2 años)
        - Nivel Medio (2-4 años)
        - Transición (4-6 años)
        - 1° a 6° Básico
        - 7° básico a 2° medio
        - 3° a 4° medio

        Responde en formato JSON:
        {
            "tipo_contenido": "PLANIFICACIÓN/EVALUACIÓN/GUÍA",
            "asignatura": "nombre_asignatura o null si no se menciona",
            "nivel": "nivel_educativo o null si no se menciona",
            "informacion_faltante": ["asignatura" y/o "nivel" si falta alguno]
        }"""),
        HumanMessage(content=query)
    ]

    interpretation = rate_limited_llm_call(llm.invoke, interpret_prompt)

    try:
        # Verificar si falta información esencial
        info = eval(interpretation.content)
        if info.get("informacion_faltante"):
            faltantes = info["informacion_faltante"]
            preguntas = {
                "asignatura": "¿Para qué asignatura necesitas este contenido?",
                "nivel": "¿Para qué nivel educativo necesitas este contenido? (Por ejemplo: 2° básico, 8° básico, etc.)"
            }

            mensaje_solicitud = "Para generar el contenido, necesito algunos detalles adicionales:\n\n"
            for faltante in faltantes:
                mensaje_solicitud += f"- {preguntas[faltante]}\n"

            return mensaje_solicitud
    except Exception as e:
        print(f"Error al procesar la interpretación: {e}")

    # Si tenemos toda la información necesaria, continuamos con la generación del contenido
    if not documents:
        return "No he podido encontrar información relevante para responder a tu pregunta."

    # Format context from documents
    context = "\n\n".join([doc.page_content for doc in documents])

    # Get source information for citation
    sources = []
    if source_documents:
        for doc in source_documents:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source = doc.metadata['source'].split('\\')[-1]
                if source not in sources:
                    sources.append(source)

    source_text = ", ".join(
        sources) if sources else "los documentos disponibles"

    # Format conversation history if available
    history_text = ""
    if conversation_history and len(conversation_history) > 0:
        history_messages = []
        for msg in conversation_history[-4:]:
            if isinstance(msg, HumanMessage):
                history_messages.append(f"Usuario: {msg.content}")
            elif isinstance(msg, AIMessage):
                history_messages.append(f"Asistente: {msg.content}")
        history_text = "\n".join(history_messages)

    # Create prompt for answer generation based on content type
    system_prompt = f"""Eres un asistente especializado para docentes chilenos que GENERA contenido educativo basándose
    en los documentos curriculares oficiales almacenados en nuestra base de datos.

    Tu tarea es CREAR uno de estos tres tipos de contenido, según lo que solicite el docente:

    Si es PLANIFICACIÓN:
    1. Objetivo general (extraído de las bases curriculares)
    2. Objetivos específicos (mínimo 5, basados en el currículum nacional)
    3. Contenidos y habilidades asociadas (mínimo 5, según nivel y asignatura)
    4. Actividades sugeridas (mínimo 3, adaptadas al contexto chileno)

    Si es EVALUACIÓN:
    1. 8 preguntas de selección múltiple con 4 opciones cada una
       - Basadas en los objetivos de aprendizaje del currículum
       - Opciones coherentes con el nivel educativo
    2. 2 preguntas de desarrollo que evalúen habilidades superiores
    3. Respuestas guía fundamentadas en el currículum nacional

    Si es GUÍA DE ESTUDIO:
    1. Resumen del tema alineado con el currículum
    2. Conceptos clave según las bases curriculares
    3. Ejemplos resueltos contextualizados a la realidad chilena
    4. Ejercicios de práctica graduados por dificultad

    IMPORTANTE:
    - NO esperes que el usuario te proporcione el contenido
    - GENERA el contenido basándote en los documentos curriculares de nuestra base de datos
    - ADAPTA el contenido al nivel y asignatura solicitados
    - CITA las fuentes curriculares específicas que utilizaste
    - Si no encuentras información suficiente en la base de datos, GENERA una alternativa razonable
      basada en el currículum nacional, indicando claramente que es una sugerencia
"""

    messages = [
        SystemMessage(content=system_prompt)
    ]

    if history_text:
        messages.append(SystemMessage(content=f"""
        Historial reciente de la conversación:
        {history_text}
        """))

    messages.append(HumanMessage(content=f"""
    SOLICITUD DEL USUARIO: {query}
    INTERPRETACIÓN: {interpretation.content}

    INSTRUCCIÓN: Basándote en los documentos curriculares disponibles en el contexto,
    CREA el contenido solicitado. NO evalúes contenido existente, GENERA uno nuevo.

    Contexto disponible:
    {context}
    """))

    try:
        answer_result = rate_limited_llm_call(llm.invoke, messages)
        answer_text = answer_result.content.strip()

        # Format final output
        final_response = f"""Basado en {source_text}, aquí está el contenido solicitado:

{answer_text}"""
        return final_response

    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"Lo siento, hubo un error al procesar la respuesta. ¿Podrías reformular tu pregunta? Error: {str(e)}"

def create_enhanced_retriever_tool(vectorstore, llm, conversation_history=None,
                               tool_name="enhanced_pdf_retriever",
                               tool_description="Busca información específica en documentos curriculares chilenos para ayudar a crear planificaciones educativas."):
    """
    Crea una herramienta mejorada que considera el historial de la conversación.
    """
    from langchain.tools import Tool

    def enhanced_retriever_with_answer(query: str) -> str:
        # 1. Primero, buscar en bases curriculares
        curriculum_docs = vectorstore.max_marginal_relevance_search(
            query=query,
            k=3,
            filter={"source": {"$contains": "bases_curriculares"}},
            fetch_k=6,
            lambda_mult=0.7
        )

        # 2. Luego, buscar en orientaciones didácticas
        orientation_docs = vectorstore.max_marginal_relevance_search(
            query=query,
            k=2,
            filter={"source": {"$contains": "orientaciones"}},
            fetch_k=4,
            lambda_mult=0.7
        )

        # 3. Finalmente, buscar en ejemplos y actividades
        activity_docs = vectorstore.max_marginal_relevance_search(
            query=query,
            k=2,
            filter={"source": {"$contains": "actividades"}},
            fetch_k=4,
            lambda_mult=0.7
        )

        # Combinar resultados priorizando bases curriculares
        all_docs = curriculum_docs + orientation_docs + activity_docs
        
        return direct_answer_generator(llm, query, all_docs, all_docs, conversation_history)

    return Tool(
        name=tool_name,
        description=tool_description,
        func=enhanced_retriever_with_answer,
    )

def create_contextual_retriever_tool(vectorstore, llm, conversation_history=None,
                         tool_name="contextual_retriever",
                         tool_description="Encuentra información amplia del currículum nacional chileno para fundamentar planificaciones educativas completas."):
    """
    Crea una herramienta para búsquedas basadas en el contexto de la conversación.
    """
    from langchain.tools import Tool
    
    def contextual_retriever_with_answer(query: str) -> str:
        print(f"Ejecutando búsqueda contextual para: {query}")

        # If we have conversation history, use it to enhance the query
        if conversation_history and len(conversation_history) > 0:
            # Extract last 2 exchanges (up to 4 messages)
            recent_history = conversation_history[-4:] if len(
                conversation_history) >= 4 else conversation_history

            # Format as text for context
            history_text = "\n".join([msg.content for msg in recent_history])

            # Create a context-aware search prompt
            context_prompt = SystemMessage(content="""
            Como experto en currículum chileno, tu objetivo es reformular la consulta para obtener información contextual completa que ayude al docente a:

            1. Comprender marcos curriculares completos por nivel educativo (desde sala cuna hasta educación media)
            2. Identificar conexiones entre asignaturas y objetivos de aprendizaje transversales
            3. Fundamentar planificaciones anuales o mensuales con criterios oficiales

            Reformula la consulta para extraer información curricular amplia y contextualizada.
            """)

            try:
                enhanced_query_response = rate_limited_llm_call(
                    llm.invoke, context_prompt.format_messages())
                enhanced_query = enhanced_query_response.content.strip()
                print(f"Consulta mejorada: {enhanced_query}")

                # Use the enhanced query
                retrieved_docs = vectorstore.similarity_search(
                    query=enhanced_query,
                    k=5  # Get more docs for contextual understanding
                )
            except Exception as e:
                print(f"Error al mejorar la consulta: {e}")
                # Fallback to original query
                retrieved_docs = vectorstore.similarity_search(
                    query=query,
                    k=4
                )
        else:
            # No history available, use standard search
            retrieved_docs = vectorstore.similarity_search(
                query=query,
                k=4
            )

        # Generate response
        return direct_answer_generator(llm, query, retrieved_docs, retrieved_docs, conversation_history)

    return Tool(
        name=tool_name,
        description=tool_description,
        func=contextual_retriever_with_answer,
    )

def create_strategic_retriever(vectorstore, query, filters=None):
    """
    Búsqueda estratégica en la base de datos vectorial
    """
    results = []
    
    # 1. Búsqueda por relevancia (MMR)
    mmr_results = vectorstore.max_marginal_relevance_search(
        query=query,
        k=5,
        fetch_k=10,
        lambda_mult=0.7,
        filter=filters
    )
    results.extend(mmr_results)
    
    # 2. Búsqueda por similitud semántica
    similarity_results = vectorstore.similarity_search(
        query=query,
        k=3,
        filter=filters
    )
    results.extend(similarity_results)
    
    # 3. Eliminar duplicados manteniendo el orden
    seen = set()
    unique_results = []
    for doc in results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_results.append(doc)
    
    return unique_results

def create_langgraph_agent(llm, tools):
    """
    Creates a LangGraph-based agent with robust memory management.
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph, MessagesState
    from langgraph.graph.message import add_messages
    from langchain_core.messages import trim_messages
    
    workflow = StateGraph(MessagesState)

    # Define trimmer para mantener un contexto manejable
    trimmer = trim_messages(strategy="last", max_tokens=16, token_counter=len)

    def call_model(state: MessagesState) -> dict:
        """Process the messages with conversation history."""
        messages = state["messages"]

        # Sistema base y contexto
        system_prompt = SystemMessage(content="""
        Eres un asistente experto en análisis de documentos curriculares chilenos.
        Tienes acceso al historial de la conversación y debes usarlo para proporcionar respuestas contextualizadas.
        """)

        # Usar el trimmer para mantener solo los mensajes relevantes
        context_messages = [system_prompt, *trimmer.invoke(messages)]

        try:
            # Encontrar la herramienta apropiada
            tool = next((t for t in tools if t.name ==
                        "enhanced_pdf_retriever"), None)
            if not tool:
                return {
                    "messages": add_messages(
                        messages,
                        AIMessage(
                            content="Error: No se encontró la herramienta de búsqueda.")
                    )
                }

            # Extraer la última pregunta
            user_question = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    user_question = msg.content
                    break

            if not user_question:
                return {
                    "messages": add_messages(
                        messages,
                        AIMessage(
                            content="No entendí tu pregunta. ¿Puedes reformularla?")
                    )
                }

            # Usar la herramienta con el contexto
            result = tool.func(user_question)

            # Generar respuesta
            final_prompt = [
                *context_messages,
                HumanMessage(content=f"""
                Basándote en la información encontrada:
                {result}

                Proporciona una respuesta clara y útil.
                """)
            ]

            final_response = rate_limited_llm_call(llm.invoke, final_prompt)

            return {
                "messages": add_messages(
                    messages,
                    AIMessage(content=final_response.content)
                )
            }

        except Exception as e:
            print(f"Error en call_model: {e}")
            return {
                "messages": add_messages(
                    messages,
                    AIMessage(
                        content=f"Hubo un error al procesar tu pregunta: {str(e)}")
                )
            }

    # Configurar el grafo
    workflow.add_node("call_model", call_model)
    workflow.set_entry_point("call_model")
    workflow.add_edge("call_model", END)

    # Crear el memory saver y compilar
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app