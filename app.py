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
credentials_path = r"C:/Users/Dante/rag_docente_sin_UI/db/gen-lang-client-0115469242-239dc466873d.json"
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


def create_filtered_retriever(vectorstore, filter_text, k=2):
    """
    Crea un retriever que filtra los resultados después de la búsqueda
    """
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k * 2}  # Duplicamos k para compensar el filtrado posterior
    )
    
    def filtered_search(query):
        docs = base_retriever.invoke(query)
        # Filtrar documentos que contengan el texto especificado en su source
        filtered_docs = [
            doc for doc in docs 
            if hasattr(doc, 'metadata') 
            and 'source' in doc.metadata 
            and filter_text.lower() in doc.metadata['source'].lower()
        ]
        return filtered_docs[:k]  # Devolver solo los k primeros documentos
    
    return filtered_search


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
        legal_docs = create_filtered_retriever(vectorstore, "leyes")(legal_query)
        if legal_docs:
            results.append("MARCO NORMATIVO:")
            for doc in legal_docs:
                results.append(doc.page_content[:500] + "...")

        # Paso 2: Buscar en orientaciones
        orientation_query = f"orientaciones para planificación en {nivel} {asignatura}"
        orientation_docs = create_filtered_retriever(vectorstore, "orientaciones")(orientation_query)
        if orientation_docs:
            results.append("\nORIENTACIONES:")
            for doc in orientation_docs:
                results.append(doc.page_content[:500] + "...")

        # Paso 3: Buscar en bases curriculares
        curriculum_query = f"objetivos de aprendizaje {nivel} {asignatura}"
        curriculum_docs = create_filtered_retriever(vectorstore, "bases curriculares")(curriculum_query)
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
       - Partir de conocimientos básicos/diagnóstico
       - Avanzar gradualmente en complejidad
       - Conectar contenidos entre meses
       - Reforzar aprendizajes previos
       - Introducir nuevos desafíos progresivamente

    3. ESTRUCTURA DE LA PLANIFICACIÓN:
       - Objetivo general (del currículum nacional)
       - Objetivos específicos (mínimo 5)
       - Contenidos y habilidades (mínimo 5)
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

    IMPORTANTE: 
    - Asegura que cada mes construya sobre el anterior
    - Incluye actividades contextualizadas al mes
    - Considera el clima y estación del año
    - Incorpora eventos relevantes del calendario escolar
    """

    def planning_agent_executor(query, asignatura=None, nivel=None):
        # Verificar información faltante
        faltante = []
        if not asignatura:
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

        # Extraer mes si se menciona
        extract_month_prompt = [
            SystemMessage(content="Extrae el mes mencionado en esta solicitud (marzo a diciembre). Si no hay ninguno, responde 'No especificado'."),
            HumanMessage(content=query)
        ]
        mes_result = rate_limited_llm_call(llm.invoke, extract_month_prompt)
        mes = mes_result.content.strip() if "No especificado" not in mes_result.content else None

        # Buscar información curricular base
        curriculum_query = f"objetivos de aprendizaje {asignatura} {nivel}"
        curriculum_docs = create_filtered_retriever(vectorstore, "bases curriculares")(curriculum_query)
        curriculum_context = "\n\n".join([doc.page_content for doc in curriculum_docs])

        # Buscar información de progresión
        progression_query = f"progresión aprendizaje {asignatura} {nivel}"
        progression_docs = create_filtered_retriever(vectorstore, "progresión")(progression_query)
        progression_context = "\n\n".join([doc.page_content for doc in progression_docs])

        # Generar la planificación
        planning_prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            SOLICITUD: {query}
            ASIGNATURA: {asignatura}
            NIVEL: {nivel}
            MES ESPECÍFICO: {mes if mes else "No especificado - considerar progresión anual"}

            CONTEXTO CURRICULAR:
            {curriculum_context}

            INFORMACIÓN DE PROGRESIÓN:
            {progression_context}

            Por favor, genera una planificación que:
            1. {f'Se enfoque en el mes de {mes}' if mes else 'Muestre la progresión anual'}
            2. Considere el contexto chileno y eventos relevantes
            3. Asegure una progresión clara del aprendizaje
            4. Se adapte al nivel y asignatura específicos
            """)
        ]

        response = rate_limited_llm_call(llm.invoke, planning_prompt)
        return response.content, False, {"asignatura": asignatura, "nivel": nivel}

    return planning_agent_executor


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
       - Alinear con objetivos de aprendizaje
       - Adaptar lenguaje al nivel
       - Incluir contextos significativos
       - Evaluar diferentes habilidades
       - Permitir demostrar comprensión

    IMPORTANTE: 
    - PRIORIZAR SIEMPRE los criterios específicos del usuario
    - Usar configuración por defecto solo cuando no hay especificaciones
    - Incluir retroalimentación para cada pregunta
    - Proporcionar rúbricas detalladas
    """

    def evaluation_agent_executor(query, asignatura=None, nivel=None):
        # Verificar información faltante
        faltante = []
        if not asignatura:
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

        # Analizar criterios específicos del usuario
        criteria_prompt = [
            SystemMessage(content="""Analiza la solicitud y extrae los criterios específicos de evaluación.
            Responde en formato JSON:
            {
                "num_seleccion_multiple": número o null,
                "num_desarrollo": número o null,
                "contenidos_especificos": ["tema1", "tema2"] o [],
                "otros_criterios": ["criterio1", "criterio2"] o []
            }"""),
            HumanMessage(content=query)
        ]
        criteria_result = rate_limited_llm_call(llm.invoke, criteria_prompt)
        
        try:
            criterios = json.loads(criteria_result.content)
        except:
            criterios = {
                "num_seleccion_multiple": None,
                "num_desarrollo": None,
                "contenidos_especificos": [],
                "otros_criterios": []
            }

        # Buscar información curricular relevante
        curriculum_query = f"objetivos de aprendizaje evaluación {asignatura} {nivel}"
        if criterios["contenidos_especificos"]:
            curriculum_query += " " + " ".join(criterios["contenidos_especificos"])
        
        curriculum_docs = create_filtered_retriever(vectorstore, "bases curriculares")(curriculum_query)
        curriculum_context = "\n\n".join([doc.page_content for doc in curriculum_docs])

        # Generar la evaluación
        evaluation_prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            SOLICITUD: {query}
            ASIGNATURA: {asignatura}
            NIVEL: {nivel}

            CRITERIOS ESPECÍFICOS:
            - Preguntas selección múltiple: {criterios["num_seleccion_multiple"] if criterios["num_seleccion_multiple"] else "usar configuración por defecto"}
            - Preguntas desarrollo: {criterios["num_desarrollo"] if criterios["num_desarrollo"] else "usar configuración por defecto"}
            - Contenidos específicos: {", ".join(criterios["contenidos_especificos"]) if criterios["contenidos_especificos"] else "no especificados"}
            - Otros criterios: {", ".join(criterios["otros_criterios"]) if criterios["otros_criterios"] else "no especificados"}

            CONTEXTO CURRICULAR:
            {curriculum_context}

            Por favor, genera una evaluación que:
            1. Priorice los criterios específicos mencionados arriba
            2. Use la configuración por defecto para lo no especificado
            3. Se alinee con el currículum nacional
            4. Sea apropiada para el nivel y asignatura
            """)
        ]

        response = rate_limited_llm_call(llm.invoke, evaluation_prompt)
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
    Crea un agente router que actúa como coordinador principal del sistema.
    """
    system_prompt = """Eres el coordinador principal de un sistema educativo especializado para docentes chilenos.
    Tu rol es DUAL:

    1. ASISTENTE CONVERSACIONAL:
       - Respondes preguntas sobre las capacidades del sistema
       - Guías la conversación hacia los servicios educativos
       - Mantienes un tono profesional y empático

    2. COORDINADOR DE AGENTES:
       Cuando detectas una solicitud específica de contenido:
       - Identificas si es PLANIFICACIÓN, EVALUACIÓN o GUÍA
       - Verificas ASIGNATURA y NIVEL EDUCATIVO
       - Derivas al agente especializado correspondiente

    IMPORTANTE: Analiza cuidadosamente cada consulta para determinar:
    1. Si es una pregunta general sobre el sistema
    2. Si es una solicitud específica de contenido
    3. Si contiene la información necesaria (asignatura y nivel)
    """
    
    def router_execute(query, asignatura=None, nivel=None):
        try:
            # Primer análisis: Determinar el tipo de consulta y extraer información
            analysis_prompt = [
                SystemMessage(content="""Analiza la consulta y responde EXACTAMENTE en este formato JSON:
                {
                    "es_consulta_general": true/false,
                    "tipo_contenido": "PLANIFICACION"/"EVALUACION"/"GUIA"/"null",
                    "asignatura": "nombre_asignatura",
                    "nivel": "nivel_educativo"
                }"""),
                HumanMessage(content=query)
            ]
            
            analysis_result = rate_limited_llm_call(llm.invoke, analysis_prompt)
            
            # Limpiar y validar el JSON
            try:
                # Encontrar el primer { y último }
                content = analysis_result.content
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = content[start:end]
                    analysis = json.loads(json_str)
                else:
                    # Valores por defecto si no se puede parsear
                    analysis = {
                        "es_consulta_general": True,
                        "tipo_contenido": "null",
                        "asignatura": "null",
                        "nivel": "null"
                    }
            except json.JSONDecodeError:
                # Valores por defecto en caso de error
                analysis = {
                    "es_consulta_general": True,
                    "tipo_contenido": "null",
                    "asignatura": "null",
                    "nivel": "null"
                }
            
            # Si es una consulta general sobre el sistema
            if analysis["es_consulta_general"]:
                return """¡Hola! Soy tu asistente educativo especializado. Puedo ayudarte a crear:

1. PLANIFICACIONES de clase: Organizadas por mes y alineadas con el currículum
2. EVALUACIONES: Personalizadas según tus necesidades
3. GUÍAS de estudio: Material de apoyo para tus estudiantes

¿Qué tipo de contenido necesitas?""", False, {"asignatura": None, "nivel": None}, "CONVERSACION"
            
            # Si es una solicitud específica, procesar según el tipo
            tipo = analysis["tipo_contenido"]
            asignatura_detectada = None if analysis["asignatura"] == "null" else analysis["asignatura"]
            nivel_detectado = None if analysis["nivel"] == "null" else analysis["nivel"]
            
            # Usar valores proporcionados o detectados
            asignatura_final = asignatura or asignatura_detectada
            nivel_final = nivel or nivel_detectado
            
            # Si detectamos un tipo específico de contenido
            if tipo and tipo != "null":
                # Si falta información necesaria
                if not asignatura_final or not nivel_final:
                    response = "Para generar el contenido, necesito saber:\n\n"
                    if not asignatura_final:
                        response += "- ¿Para qué asignatura? (Ej: Matemáticas, Lenguaje, etc.)\n"
                    if not nivel_final:
                        response += "- ¿Para qué nivel educativo? (Ej: 2° básico, 8° básico, etc.)\n"
                    return response, True, {"asignatura": asignatura_final, "nivel": nivel_final}, tipo
                
                # Si tenemos toda la información, derivar al agente correspondiente
                if tipo == "GUIA":
                    response, _, _ = study_guide_agent(query, asignatura_final, nivel_final)
                elif tipo == "EVALUACION":
                    response, _, _ = evaluation_agent(query, asignatura_final, nivel_final)
                elif tipo == "PLANIFICACION":
                    response, _, _ = planning_agent(query, asignatura_final, nivel_final)
                else:
                    response = "No pude identificar el tipo de contenido que necesitas. ¿Podrías especificar si quieres una planificación, evaluación o guía de estudio?"
                
                return response, False, {"asignatura": asignatura_final, "nivel": nivel_final}, tipo
            
            # Si no se detectó el tipo de contenido
            return """Por favor, especifica qué tipo de contenido necesitas:
1. PLANIFICACIÓN
2. EVALUACIÓN
3. GUÍA DE ESTUDIO""", True, {"asignatura": None, "nivel": None}, None

        except Exception as e:
            print(f"\n⚠️ Error en router: {str(e)}")
            return """Disculpa, hubo un error al procesar tu solicitud. 
            ¿Podrías especificar claramente:
            1. Qué tipo de contenido necesitas (planificación, evaluación o guía)
            2. Para qué asignatura
            3. Para qué nivel educativo""", True, {"asignatura": None, "nivel": None}, None
    
    return router_execute

# Modificación de la función main para implementar arquitectura multi-agente


def main():
    print("\n" + "=" * 50)
    print("🎓 Sistema Educativo Multi-Agente")
    print("=" * 50)

    # Configurar el LLM
    print("\n⚙️ Inicializando componentes...")
    llm = ChatVertexAI(
        model_name="gemini-1.5-flash",
        temperature=0.5,
        max_output_tokens=8192,
        top_p=0.95,
        top_k=40,
    )

    embeddings = VertexAIEmbeddings(model_name="text-multilingual-embedding-002")

    # Inicializar vectorstore
    try:
        collection_name = "pdf-rag-chroma"
        persist_directory = f"./{collection_name}"

        if os.path.exists(persist_directory):
            print("\n📚 Cargando base de datos existente...")
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=collection_name,
            )
            collection_size = len(vectorstore.get()["ids"])
            print(f"✅ Base de datos cargada con {collection_size} documentos")
        else:
            print("\n📄 Cargando documentos PDF...")
            pdf_directory = "pdf_docs"
            if not os.path.exists(pdf_directory):
                os.makedirs(pdf_directory)
                print(f"📁 Directorio creado: {pdf_directory}")
                print("❌ Coloca archivos PDF en el directorio y reinicia el programa.")
                return

            documents = load_pdf_documents(pdf_directory)
            if not documents:
                print("❌ No se encontraron documentos PDF.")
                return

            print(f"✅ {len(documents)} páginas cargadas")
            print("\n🔄 Creando base de datos...")
            vectorstore = create_vectorstore(documents, embeddings, collection_name)

    except Exception as e:
        print(f"\n❌ Error al inicializar la base de datos: {e}")
        return

    # Crear agentes especializados
    print("\n🤖 Inicializando agentes especializados...")
    planning_agent = create_planning_agent(llm, vectorstore)
    print("✅ Agente de Planificación listo")

    evaluation_agent = create_evaluation_agent(llm, vectorstore)
    print("✅ Agente de Evaluación listo")

    study_guide_agent = create_study_guide_agent(llm, vectorstore)
    print("✅ Agente de Guías de Estudio listo")

    # Crear agente router
    print("\n🧭 Inicializando agente coordinador...")
    router = create_router_agent(
        llm, planning_agent, evaluation_agent, study_guide_agent
    )
    print("✅ Agente Coordinador listo")

    print("\n" + "=" * 50)
    print("🎯 Sistema listo para procesar solicitudes!")
    print("Puedes solicitar:")
    print("1. PLANIFICACIONES educativas")
    print("2. EVALUACIONES")
    print("3. GUÍAS de estudio")
    print("=" * 50)

    # Generar ID de sesión
    thread_id = str(uuid.uuid4())[:8]
    print(f"\n🔑 ID de sesión: {thread_id}")

    # Estado de la sesión
    session_state = {
        "pending_request": False,
        "last_query": "",
        "asignatura": None,
        "nivel": None,
        "tipo": None,
    }

    while True:
        try:
            query = input("\n👤 Usuario: ").strip()
            
            if query.lower() in ["exit", "quit", "q", "salir"]:
                print("\n👋 ¡Hasta luego!")
                break

            if not query:
                print("❌ Por favor, ingresa una consulta válida.")
                continue

            # Reiniciar el estado para cada nueva consulta completa
            if not session_state["pending_request"]:
                session_state = {
                    "pending_request": False,
                    "last_query": "",
                    "asignatura": None,
                    "nivel": None,
                    "tipo": None
                }

            # Procesar la consulta a través del router
            if session_state["pending_request"]:
                # Completando información faltante
                if not session_state["asignatura"]:
                    session_state["asignatura"] = query
                    print("\n🔄 Procesando...")
                    response, needs_info, info, tipo = router(
                        session_state["last_query"], 
                        session_state["asignatura"], 
                        session_state["nivel"]
                    )
                elif not session_state["nivel"]:
                    session_state["nivel"] = query
                    print("\n🔄 Procesando...")
                    response, needs_info, info, tipo = router(
                        session_state["last_query"], 
                        session_state["asignatura"], 
                        session_state["nivel"]
                    )
            else:
                # Nueva consulta - siempre pasa por el router
                print("\n🔄 Analizando tu solicitud...")
                response, needs_info, info, tipo = router(query)
                
            # Manejar la respuesta
            if needs_info:
                print(f"\n❓ {response}")
                session_state.update({
                    "pending_request": True,
                    "last_query": query if not session_state["last_query"] else session_state["last_query"],
                    "asignatura": info.get("asignatura"),
                    "nivel": info.get("nivel"),
                    "tipo": tipo
                })
            else:
                print(f"\n🤖 {response}")
                # Guardar la conversación
                full_query = query
                if session_state["asignatura"] or session_state["nivel"]:
                    full_query = f"{query} (Asignatura: {session_state['asignatura']}, Nivel: {session_state['nivel']})"
                format_and_save_conversation(full_query, response, thread_id)
                # Reiniciar estado completamente después de cada respuesta completa
                session_state = {
                    "pending_request": False,
                    "last_query": "",
                    "asignatura": None,
                    "nivel": None,
                    "tipo": None
                }

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Por favor, intenta reformular tu solicitud.")
            # Reiniciar estado en caso de error
            session_state = {
                "pending_request": False,
                "last_query": "",
                "asignatura": None,
                "nivel": None,
                "tipo": None
            }


if __name__ == "__main__":
    main()
