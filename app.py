"""from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory"""

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.graph.message import add_messages

import os
import sys
import time
import uuid
from ratelimit import limits, sleep_and_retry
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, TypedDict
import json


# Configure stdout to handle special characters properly
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), 'db', '.env')
load_dotenv(dotenv_path)

# Credenciales para usar VERTEX_AI
credentials_path = r"C:/Users/Dante/Desktop/rag_docente_sin_UI/db/gen-lang-client-0115469242-239dc466873d.json"
if not os.path.exists(credentials_path):
    raise FileNotFoundError(
        f"No se encontró el archivo de credenciales en: {credentials_path}")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
os.environ["LANGSMITH_TRACING"] = "true"

# Para tracear con langsmith
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if not LANGSMITH_API_KEY:
    raise ValueError(
        "La variable LANGSMITH_API_KEY no está definida en el archivo .env en la carpeta db/")
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY

# Definir el límite de peticiones (por ejemplo, 180 por minuto para estar seguros)
CALLS_PER_MINUTE = 150
PERIOD = 60
WAIT_TIME = 1


@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=PERIOD)
def rate_limited_llm_call(func, *args, **kwargs):
    """
    Wrapper function para las llamadas al LLM con rate limiting mejorado
    """
    time.sleep(WAIT_TIME)
    return func(*args, **kwargs)

# Iniciamos con el programa


def load_pdf_documents(directory_path: str) -> List:
    """
    Loads all PDF documents from a given directory and its subdirectories recursively.

    Args:
        directory_path: Path to the directory containing PDF files

    Returns:
        List of Document objects containing the content of the PDFs
    """
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return []

    documents = []

    # Recorrer el directorio y sus subdirectorios
    for root, dirs, files in os.walk(directory_path):
        pdf_files = [f for f in files if f.endswith('.pdf')]

        for pdf_file in pdf_files:
            try:
                file_path = os.path.join(root, pdf_file)
                print(f"Loading PDF: {file_path}")
                # PyPDFLoader handles both text and images
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                print(
                    f"Successfully loaded {pdf_file} with {len(loader.load())} pages")
            except Exception as e:
                print(f"Error loading {pdf_file}: {e}")

    if not documents:
        print(f"No PDF files found in {directory_path} or its subdirectories.")

    return documents


def create_vectorstore(documents, embeddings, collection_name="pdf-rag-chroma"):
    """
    Crea o carga un Chroma vector store.
    Si la base de datos existe, simplemente la carga sin procesar nuevos documentos.
    Solo crea una nueva si no existe.
    """
    persist_directory = f"./{collection_name}"

    # Si existe la base de datos, cargarla y retornar
    if os.path.exists(persist_directory):
        print(
            f"\n📚 Base de datos Chroma existente encontrada en {persist_directory}")
        try:
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            collection_size = len(vectorstore.get()['ids'])
            print(
                f"✅ Vectorstore cargado exitosamente con {collection_size} documentos")
            return vectorstore
        except Exception as e:
            print(f"❌ Error crítico al cargar la base de datos existente: {e}")
            raise e

    # Solo si NO existe la base de datos, crear una nueva
    print("\n⚠️ No se encontró una base de datos existente. Creando nueva...")

    if not documents:
        raise ValueError(
            "❌ No se proporcionaron documentos para crear el vectorstore")

    print("📄 Procesando documentos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Documentos divididos en {len(chunks)} chunks")

    print("🔄 Creando nuevo vectorstore...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    collection_size = len(vectorstore.get()['ids'])
    print(f"✅ Nuevo vectorstore creado con {collection_size} documentos")

    return vectorstore


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
    def enhanced_retriever_with_answer(query: str) -> str:
        print(f"Ejecutando búsqueda mejorada para: {query}")

        # Si hay historial, usarlo para mejorar la búsqueda
        if conversation_history and len(conversation_history) > 0:
            # Crear un prompt para reformular la consulta
            context_prompt = SystemMessage(content="""
            Eres un especialista en educación chilena. Tu tarea es reformular la consulta del usuario para extraer información relevante de documentos curriculares que le ayude a crear:

            1. Planificaciones educativas para niveles desde sala cuna hasta educación media
            2. Actividades adaptadas al currículum nacional
            3. Evaluaciones alineadas con objetivos de aprendizaje oficiales

            Basado en el historial de la conversación y la pregunta actual, reformula la consulta para encontrar información curricular específica que satisfaga la necesidad del docente.
            """)

            try:
                # Reformular la consulta
                enhanced_query_response = rate_limited_llm_call(
                    llm.invoke, context_prompt)
                enhanced_query = enhanced_query_response.content
                print(f"Consulta mejorada: {enhanced_query}")

                # Realizar la búsqueda con la consulta mejorada
                retrieved_docs = vectorstore.max_marginal_relevance_search(
                    query=enhanced_query,
                    k=6,
                    fetch_k=10,
                    lambda_mult=0.7
                )
            except Exception as e:
                print(f"Error al mejorar la consulta: {e}")
                retrieved_docs = vectorstore.max_marginal_relevance_search(
                    query=query,
                    k=6,
                    fetch_k=10,
                    lambda_mult=0.7
                )
        else:
            retrieved_docs = vectorstore.max_marginal_relevance_search(
                query=query,
                k=6,
                fetch_k=10,
                lambda_mult=0.7
            )

        return direct_answer_generator(llm, query, retrieved_docs, retrieved_docs, conversation_history)

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


def create_strategic_search_tool(vectorstore, llm, conversation_history=None,
                               tool_name="strategic_curriculum_search",
                               tool_description="Realiza una búsqueda estratégica en todos los recursos curriculares siguiendo un orden específico para planificaciones completas."):

    context_prompt = SystemMessage(content="""
    Como especialista en currículum chileno, sigue este proceso de búsqueda estratégica:

    1. LEYES: Identifica los requisitos normativos aplicables
    2. ORIENTACIONES: Comprende la estructura recomendada
    3. BASES CURRICULARES: Encuentra objetivos específicos por nivel y asignatura
    4. PROPUESTAS: Busca planificaciones existentes similares
    5. ACTIVIDADES SUGERIDAS: Complementa con actividades concretas

    Reformula la consulta del usuario para encontrar información siguiendo este orden preciso.
    """)

    def strategic_search_with_answer(query: str) -> str:
        # Extraer nivel y asignatura de la consulta
        extraction_prompt = [
            context_prompt,
            HumanMessage(
                content=f"Extrae el nivel educativo y asignatura de esta consulta: '{query}'. Responde solo con el formato 'NIVEL: X, ASIGNATURA: Y'.")
        ]
        extraction_response = rate_limited_llm_call(
            llm.invoke, extraction_prompt).content

        try:
            nivel = extraction_response.split(
                "NIVEL:")[1].split(",")[0].strip()
            asignatura = extraction_response.split("ASIGNATURA:")[1].strip()
        except:
            nivel = ""
            asignatura = ""

        results = []

        # Paso 1: Buscar en leyes
        legal_query = f"requisitos normativos para planificaciones educativas en {nivel} {asignatura}"
        legal_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={
                                                   "k": 2, "filter": {"source": {"$contains": "leyes"}}})
        legal_docs = legal_retriever.invoke(legal_query)
        if legal_docs:
            results.append("MARCO NORMATIVO:")
            for doc in legal_docs:
                results.append(doc.page_content[:500] + "...")

        # Paso 2: Buscar en orientaciones
        orientation_query = f"orientaciones para planificación en {nivel} {asignatura}"
        orientation_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={
                                                         "k": 2, "filter": {"source": {"$contains": "orientaciones"}}})
        orientation_docs = orientation_retriever.invoke(
            orientation_query)
        if orientation_docs:
            results.append("\nORIENTACIONES:")
            for doc in orientation_docs:
                results.append(doc.page_content[:500] + "...")

        # Paso 3: Buscar en bases curriculares
        curriculum_query = f"objetivos de aprendizaje {nivel} {asignatura}"
        curriculum_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={
                                                        "k": 3, "filter": {"source": {"$contains": "bases curriculares"}}})
        curriculum_docs = curriculum_retriever.invoke(
            curriculum_query)
        if curriculum_docs:
            results.append("\nBASES CURRICULARES:")
            for doc in curriculum_docs:
                results.append(doc.page_content[:500] + "...")

        # Paso 4: Buscar en propuestas
        proposal_query = f"propuesta planificación {nivel} {asignatura} {query}"
        proposal_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={
                                                      "k": 2, "filter": {"source": {"$contains": "propuesta"}}})
        proposal_docs = proposal_retriever.invoke(
            proposal_query)
        if proposal_docs:
            results.append("\nPROPUESTAS EXISTENTES:")
            for doc in proposal_docs:
                results.append(doc.page_content[:500] + "...")

        # Paso 5: Buscar en actividades sugeridas
        activity_query = f"actividades sugeridas {nivel} {asignatura} {query}"
        activity_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={
                                                      "k": 3, "filter": {"source": {"$contains": "actividades sugeridas"}}})
        activity_docs = activity_retriever.invoke(
            activity_query)
        if activity_docs:
            results.append("\nACTIVIDADES SUGERIDAS:")
            for doc in activity_docs:
                results.append(doc.page_content[:500] + "...")

        if not results:
            return "No se encontró información específica siguiendo el proceso de búsqueda estratégica."

        return "\n".join(results)

    return Tool(
        name=tool_name,
        func=strategic_search_with_answer,
        description=tool_description
    )


def create_agent(llm, tools):
    """
    Creates a LangChain agent with an improved prompt that emphasizes
    contextual understanding and helpfulness over precision.
    """
    # Obtener nombres de herramientas
    tool_names = ", ".join([tool.name for tool in tools])

    # Define the prompt with emphasis on being helpful with available information
    system_message = f"""Eres un asistente especializado para docentes chilenos que ayuda en la creación de tres tipos específicos de contenido educativo:

    1. PLANIFICACIONES:
       - Anuales, semestrales o mensuales
       - Incluyen: objetivo general, objetivos específicos (mínimo 5), contenidos y habilidades asociadas (mínimo 5)
       - Adaptadas al nivel educativo solicitado

    2. EVALUACIONES:
       - 10 preguntas total:
         * 8 preguntas de selección múltiple
         * 2 preguntas de desarrollo
       - Incluye respuestas guía para todas las preguntas
       - Adaptadas al nivel y asignatura

    3. GUÍAS DE ESTUDIO:
       - Contenido específico por asignatura y tema
       - Formato de repaso estructurado y sencillo
       - Incluye ejemplos y ejercicios prácticos

    PROCESO DE RESPUESTA:
    1. INTERPRETAR la solicitud del usuario para identificar qué tipo de contenido necesita
    2. GENERAR SOLO el tipo de contenido solicitado (no generar los otros tipos)
    3. Si no hay información específica en el contexto, OFRECER una alternativa aclarando que no está basada en los documentos disponibles


    Aspectos a considerar, estos son los niveles educativos que tiene el sistema nacional chileno:
    - Niveles: Sala Cuna (0-2 años), Nivel Medio (2-4 años), Transición (4-6 años)
    - Educación Básica (1° a 6° Básico)
    - Educación Media (7° básico a 2° medio)
    - Educación Media diferenciada (3° a 4° medio. Científico-Humanista, Técnico Profesional, Artística)
    Tienes acceso a las siguientes herramientas:
    {tool_names}

    ORGANIZACIÓN DE CONTENIDOS:
    - LEYES: Marco normativo que define los límites y requisitos de planificaciones educativas
    - ORIENTACIONES: Guías generales sobre cómo estructurar planificaciones y evaluaciones
    - BASES CURRICULARES: Documentos oficiales con objetivos de aprendizaje por nivel y asignatura
    - PROPUESTAS: Ejemplos de planificaciones que puedes adaptar (prioriza usar estas antes de crear desde cero)
    - ACTIVIDADES SUGERIDAS: Actividades específicas que complementan las planificaciones

    INSTRUCCIONES IMPORTANTES:
    1. SIEMPRE identifica primero qué tipo de contenido necesita el usuario
    2. GENERA ÚNICAMENTE el tipo de contenido solicitado
    3. MANTÉN el formato específico para cada tipo de contenido
    4. Si no encuentras información específica, OFRECE alternativas aclarando que no están basadas en el contexto
    5. Sé conversacional y natural en tus respuestas

    Para usar una herramienta, utiliza el siguiente formato:
    Thought: Primero, necesito pensar qué hacer
    Action: la acción a realizar, debe ser una de [{tool_names}]
    Action Input: la entrada para la acción
    Observation: el resultado de la acción
    ... (este proceso puede repetirse si es necesario)
    Thought: Ahora conozco la respuesta final
    Final Answer: la respuesta final a la pregunta original

    Utiliza siempre el formato anterior, comenzando con un Thought e incluyendo los pasos Action/Action Input necesarios antes de proporcionar una Final Answer.

    Question: {input}
    """

    prompt = PromptTemplate(
        template=system_message + "\n\nQuestion: {input}",
        input_variables=["input", "agent_scratchpad"]
    )

    # Create the agent
    agent = create_react_agent(llm, tools, prompt)

    # Create the agent executor with improved settings
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=2,  # Limit to prevent loops
        early_stopping_method="force",  # Force stop after max iterations
        return_intermediate_steps=True  # Return the steps for tracing
    )

    return agent_executor

# Define the state schema as a TypedDict


class AgentState(TypedDict):
    messages: List[Any]  # Store conversation messages
    next_steps: List[Dict[str, str]]  # Store next actions to take
    tool_results: Dict[str, str]  # Store results from tools
    current_tool: Optional[str]  # Track which tool is being used
    conversation_summary: Optional[str]  # Store conversation summary
    thread_id: Optional[str]  # Agregamos thread_id al estado


def create_langgraph_agent(llm, tools):
    """
    Creates a LangGraph-based agent with robust memory management.
    """
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


def format_and_save_conversation(query: str, response: str, thread_id: str, output_dir: str = "conversaciones") -> str:
    """
    Formatea y guarda la conversación en un archivo Markdown usando el thread_id.
    """
    # Crear directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Usar el thread_id en el nombre del archivo
    filename = f"conversacion_N°_{thread_id}.md"
    filepath = os.path.join(output_dir, filename)

    # Si el archivo ya existe, añadir al contenido existente
    existing_content = ""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            existing_content = f.read()

    # Formatear el nuevo contenido
    new_content = f"""
## Pregunta
{query}

## Respuesta
{response}

---
"""

    # Combinar contenido existente con nuevo contenido
    markdown_content = existing_content + new_content if existing_content else f"""# Conversación RAG - Thread ID: {thread_id}
Iniciada el: {time.strftime("%d/%m/%Y %H:%M:%S")}

{new_content}"""

    # Guardar el archivo
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"\n💾 Conversación guardada en: {filepath}")
    return filepath

# ==================== AGENTES ESPECIALIZADOS ====================


def create_planning_agent(llm, vectorstore):
    """
    Crea un agente especializado en planificaciones educativas.
    Verifica que tenga asignatura y nivel antes de generar contenido.
    """
    system_prompt = """Eres un agente especializado en CREAR PLANIFICACIONES EDUCATIVAS para el sistema chileno.

    DEBES verificar que tengas estos dos datos esenciales:
    1. ASIGNATURA (Lenguaje, Matemáticas, Historia, Ciencias, etc.)
    2. NIVEL (Específico: 1° básico, 7° básico, 2° medio, etc.)

    Si falta alguno de estos datos, SOLICITA específicamente la información faltante.

    Una vez que tengas ambos datos, genera una planificación completa con:
    - Objetivo general (extraído de las bases curriculares)
    - Objetivos específicos (mínimo 5, basados en el currículum)
    - Contenidos/habilidades (mínimo 5, alineados con bases curriculares)
    - Actividades (mínimo 3, detalladas y acordes al nivel)
    - Evaluación sugerida (métodos para verificar aprendizaje)

    Formatea tu respuesta de manera clara y estructurada. Cita las fuentes curriculares específicas.
    """

    def planning_agent_executor(query, asignatura=None, nivel=None):
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

        # Si falta información, solicitarla
        if faltante:
            response = "Para crear una planificación educativa completa, necesito la siguiente información:\n\n"
            if "asignatura" in faltante:
                response += "- ¿Para qué asignatura necesitas la planificación? (Lenguaje, Matemáticas, etc.)\n"
            if "nivel" in faltante:
                response += "- ¿Para qué nivel educativo? (Ej: 2° básico, 8° básico, 3° medio, etc.)\n"
            return response, True, {"asignatura": asignatura, "nivel": nivel}

        # Si tenemos toda la información, generar la planificación
        enhanced_query = f"Crear planificación para la asignatura de {asignatura} para el nivel {nivel}"

        # Buscar información relevante en el vectorstore
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10}
        )

        retrieved_docs = retriever.invoke(enhanced_query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Generar la planificación
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Solicitud: {query}
            Asignatura: {asignatura}
            Nivel: {nivel}

            Información curricular relevante:
            {context}
            """)
        ]

        response = rate_limited_llm_call(llm.invoke, prompt)
        return response.content, False, {"asignatura": asignatura, "nivel": nivel}

    return planning_agent_executor


def create_evaluation_agent(llm, vectorstore):
    """
    Crea un agente especializado en evaluaciones educativas.
    Verifica que tenga asignatura y nivel antes de generar contenido.
    """
    system_prompt = """Eres un agente especializado en CREAR EVALUACIONES EDUCATIVAS para el sistema chileno.

    DEBES verificar que tengas estos dos datos esenciales:
    1. ASIGNATURA (Lenguaje, Matemáticas, Historia, Ciencias, etc.)
    2. NIVEL (Específico: 1° básico, 7° básico, 2° medio, etc.)

    Si falta alguno de estos datos, SOLICITA específicamente la información faltante.

    Una vez que tengas ambos datos, genera una evaluación completa con:
    - 8 preguntas de selección múltiple con 4 opciones cada una
    - 2 preguntas de desarrollo que evalúen habilidades superiores
    - Respuestas correctas y rúbricas de evaluación para las preguntas de desarrollo

    Las preguntas deben estar alineadas con los objetivos de aprendizaje del currículum nacional
    para la asignatura y nivel especificados.

    Formatea tu respuesta con números claros para cada pregunta y letras para las alternativas.
    """

    def evaluation_agent_executor(query, asignatura=None, nivel=None):
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

        # Si falta información, solicitarla
        if faltante:
            response = "Para crear una evaluación completa, necesito la siguiente información:\n\n"
            if "asignatura" in faltante:
                response += "- ¿Para qué asignatura necesitas la evaluación? (Lenguaje, Matemáticas, etc.)\n"
            if "nivel" in faltante:
                response += "- ¿Para qué nivel educativo? (Ej: 2° básico, 8° básico, 3° medio, etc.)\n"
            return response, True, {"asignatura": asignatura, "nivel": nivel}

        # Si tenemos toda la información, generar la evaluación
        enhanced_query = f"Crear evaluación para la asignatura de {asignatura} para el nivel {nivel}"

        # Buscar información relevante en el vectorstore
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10}
        )

        retrieved_docs = retriever.invoke(enhanced_query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Generar la evaluación
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Solicitud: {query}
            Asignatura: {asignatura}
            Nivel: {nivel}

            Objetivos de aprendizaje y contenidos relevantes:
            {context}
            """)
        ]

        response = rate_limited_llm_call(llm.invoke, prompt)
        return response.content, False, {"asignatura": asignatura, "nivel": nivel}

    return evaluation_agent_executor


def create_study_guide_agent(llm, vectorstore):
    """
    Crea un agente especializado en guías de estudio.
    Verifica que tenga asignatura y nivel antes de generar contenido.
    """
    system_prompt = """Eres un agente especializado en CREAR GUÍAS DE ESTUDIO para el sistema educativo chileno.

    DEBES verificar que tengas estos dos datos esenciales:
    1. ASIGNATURA (Lenguaje, Matemáticas, Historia, Ciencias, etc.)
    2. NIVEL (Específico: 1° básico, 7° básico, 2° medio, etc.)

    Si falta alguno de estos datos, SOLICITA específicamente la información faltante.

    Una vez que tengas ambos datos, genera una guía de estudio completa con:
    - Resumen del tema principal
    - Conceptos clave (definiciones claras)
    - Ejemplos resueltos paso a paso
    - Ejercicios de práctica graduados por dificultad
    - Respuestas o soluciones a los ejercicios propuestos

    La guía debe estar alineada con el currículum nacional y usar lenguaje apropiado para el nivel.

    Organiza la guía con títulos claros y formato amigable para estudiantes.
    """

    def study_guide_agent_executor(query, asignatura=None, nivel=None):
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

        # Si falta información, solicitarla
        if faltante:
            response = "Para crear una guía de estudio completa, necesito la siguiente información:\n\n"
            if "asignatura" in faltante:
                response += "- ¿Para qué asignatura necesitas la guía? (Lenguaje, Matemáticas, etc.)\n"
            if "nivel" in faltante:
                response += "- ¿Para qué nivel educativo? (Ej: 2° básico, 8° básico, 3° medio, etc.)\n"
            return response, True, {"asignatura": asignatura, "nivel": nivel}

        # Si tenemos toda la información, generar la guía
        enhanced_query = f"Crear guía de estudio para la asignatura de {asignatura} para el nivel {nivel}"

        # Buscar información relevante en el vectorstore
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10}
        )

        retrieved_docs = retriever.invoke(enhanced_query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Generar la guía
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Solicitud: {query}
            Asignatura: {asignatura}
            Nivel: {nivel}

            Contenidos y objetivos de aprendizaje relevantes:
            {context}
            """)
        ]

        response = rate_limited_llm_call(llm.invoke, prompt)
        return response.content, False, {"asignatura": asignatura, "nivel": nivel}

    return study_guide_agent_executor


def create_router_agent(llm, planning_agent, evaluation_agent, study_guide_agent):
    """
    Agente router que controla el flujo de la conversación y delega a agentes especializados.
    """
    system_prompt = """Eres un agente router que analiza solicitudes educativas.
    
    DETERMINA el tipo de contenido que necesita el usuario:
    - PLANIFICACION: si solicita planes de clase, planificaciones anuales, etc.
    - EVALUACION: si solicita pruebas, exámenes, rúbricas, etc.
    - GUIA: si solicita guías de estudio, material de repaso, etc.
    
    Responde SOLO con una de estas palabras: PLANIFICACION, EVALUACION o GUIA
    """
    
    def router_execute(query):
        """
        Analiza la consulta y delega al agente especializado apropiado.
        """
        # 1. Determinar el tipo de contenido solicitado
        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        try:
            # Obtener el tipo de contenido
            tipo_result = rate_limited_llm_call(llm.invoke, prompt)
            tipo = tipo_result.content.strip().upper()
            
            # 2. Delegar al agente especializado correspondiente
            if tipo == "PLANIFICACION":
                response, needs_info, info = planning_agent(query)
            elif tipo == "EVALUACION":
                response, needs_info, info = evaluation_agent(query)
            elif tipo == "GUIA":
                response, needs_info, info = study_guide_agent(query)
            else:
                return {
                    "status": "error",
                    "message": "No pude determinar qué tipo de contenido necesitas. ¿Podrías especificar si necesitas una planificación, evaluación o guía?"
                }
            
            # 3. Procesar la respuesta del agente especializado
            if needs_info:
                return {
                    "status": "need_info",
                    "message": response
                }
            else:
                return {
                    "status": "success",
                    "message": response
                }
                
        except Exception as e:
            print(f"Error en router_execute: {e}")
            return {
                "status": "error",
                "message": "Hubo un error al procesar tu solicitud. ¿Podrías reformularla?"
            }
    
    return router_execute

def main():
    print("Inicializando Sistema Multi-Agente Educativo...")
    
    # Configurar el LLM
    llm = ChatVertexAI(
        model_name="gemini-1.5-flash",
        temperature=0.5,
        max_output_tokens=8192,
        top_p=0.95,
        top_k=40
    )

    embeddings = VertexAIEmbeddings(model_name="text-multilingual-embedding-002")

    # Verificar si ya existe la base de datos de Chroma
    collection_name = "pdf-rag-chroma"
    persist_directory = f"./{collection_name}"

    try:
        if os.path.exists(persist_directory):
            print("\n📚 Cargando base de datos existente...")
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            collection_size = len(vectorstore.get()['ids'])
            if collection_size > 0:
                print(
                    f"✅ Base de datos cargada exitosamente con {collection_size} documentos")
            else:
                raise ValueError("La base de datos existe pero está vacía")
        else:
            # Solo cargar PDFs si necesitamos crear una nueva base de datos
            print(
                "\n📄 No se encontró base de datos existente. Cargando documentos PDF...")
            pdf_directory = "pdf_docs"
            if not os.path.exists(pdf_directory):
                os.makedirs(pdf_directory)
                print(f"📁 Se creó el directorio: {pdf_directory}")
                print("❌ Por favor, coloca archivos PDF en el directorio y reinicia el programa.")
                return

            documents = load_pdf_documents(pdf_directory)
            if not documents:
                print("❌ No se encontraron documentos PDF. Por favor, agrega archivos PDF y reinicia.")
                return

            print(f"✅ Se cargaron {len(documents)} páginas de documentos PDF")
            
            # Crear nueva base de datos
            print("\n🔄 Creando nueva base de datos...")
            vectorstore = create_vectorstore(documents, embeddings, collection_name)
            
    except Exception as e:
        print(f"\n❌ Error al inicializar la base de datos: {e}")
        return

    # Crear agentes especializados
    print("\n🤖 Creando agentes especializados...")
    planning_agent = create_planning_agent(llm, vectorstore)
    evaluation_agent = create_evaluation_agent(llm, vectorstore)
    study_guide_agent = create_study_guide_agent(llm, vectorstore)
    
    # Crear agente router
    print("🧭 Creando agente router...")
    router = create_router_agent(llm, planning_agent, evaluation_agent, study_guide_agent)

    print("\n" + "="*50)
    print("🎯 Sistema Multi-Agente Educativo listo!")
    print("Haz preguntas sobre planificaciones, evaluaciones o guías de estudio")
    print("="*50)

    # Generar un ID de sesión único
    thread_id = str(uuid.uuid4())[:8]
    print(f"\n🔑 ID de sesión: {thread_id}")

    while True:
        query = input("\n👤 Usuario: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("\n👋 Saliendo del sistema. ¡Hasta luego!")
            break

        try:
            print("\n🔄 Procesando tu solicitud...")
            result = router(query)
            
            if result["status"] == "need_info":
                print(f"\n❓ {result['message']}")
            elif result["status"] == "success":
                print(f"\n🤖 Respuesta: {result['message']}")
                # Guardar la conversación
                format_and_save_conversation(query, result['message'], thread_id)
            else:
                print(f"\n❌ {result['message']}")
                
        except Exception as e:
            print(f"\n❌ Ocurrió un error: {e}")
            print("Por favor, intenta reformular tu solicitud.")

if __name__ == "__main__":
    main()