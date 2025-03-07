import os
import sys
import uuid
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from google.cloud import aiplatform
from vertexai.language_models import ChatModel
from utils.rate_limiter import rate_limited_llm_call, CALLS_PER_MINUTE, PERIOD, WAIT_TIME
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import datetime

# Importar módulos propios - CORREGIDOS
from utils.conversation import format_and_save_conversation
from core.vectorstore.store import create_vectorstore
from core.vectorstore.retriever import create_enhanced_retriever_tool
from core.agents import (
    create_planning_agent,
    create_evaluation_agent,
    create_study_guide_agent,
    create_router_agent
)
from config import (
    PDF_DIRECTORY,
    COLLECTION_NAME,
    PERSIST_DIRECTORY,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TOP_P,
    LLM_TOP_K,
    EMBEDDING_MODEL_NAME
)
from config.env_loader import load_environment
from core.vectorstore.loader import load_pdf_documents
from core.llm.llm_config import get_llm_config, get_embedding_config

# Configuración de directorios y nombres de colección
COLLECTION_NAME = "pdf-rag-chroma"
PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", COLLECTION_NAME)
PDF_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "pdf_docs")

# Credenciales para usar VERTEX_AI
credentials_path = os.path.join(os.path.dirname(__file__), "config", "credentials", "gen-lang-client-0115469242-239dc466873d.json")
if not os.path.exists(credentials_path):
    raise FileNotFoundError(f"No se encontró el archivo de credenciales en: {credentials_path}")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
os.environ["LANGSMITH_TRACING"] = "true"

# Cargar variables de entorno de LangSmith
credentials_path_langsmith = os.path.join(os.path.dirname(__file__), "config", "credentials", ".env")
load_dotenv(credentials_path_langsmith)

# Verificar que se cargó la API key de LangSmith
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if not LANGSMITH_API_KEY:
    raise ValueError("La variable LANGSMITH_API_KEY no está definida en el archivo .env en la carpeta credentials/")
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY

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
                print(f"Successfully loaded {pdf_file} with {len(loader.load())} pages")
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
        print(f"\n📚 Base de datos Chroma existente encontrada en {persist_directory}")
        try:
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            collection_size = len(vectorstore.get()['ids'])
            print(f"✅ Vectorstore cargado exitosamente con {collection_size} documentos")
            return vectorstore
        except Exception as e:
            print(f"❌ Error crítico al cargar la base de datos existente: {e}")
            raise e

    # Solo si NO existe la base de datos, crear una nueva
    print("\n⚠️ No se encontró una base de datos existente. Creando nueva...")

    if not documents:
        raise ValueError("❌ No se proporcionaron documentos para crear el vectorstore")

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

def main():
    print("Inicializando Sistema Multi-Agente Educativo...")

    # Configurar rutas absolutas
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_directory = os.path.join(base_dir, "data", "raw", "pdf_docs")
    persist_directory = os.path.join(base_dir, "data", "processed", "pdf-rag-chroma")
    
    print(f"📁 Directorio base: {base_dir}")
    print(f"📁 Directorio de PDFs: {pdf_directory}")
    print(f"📁 Directorio de la base de datos: {persist_directory}")

    # Cargar configuración de entorno
    load_environment()

    # Obtener configuraciones
    llm_config = get_llm_config()
    embedding_config = get_embedding_config()

    # Configurar el LLM
    print("\n⚙️ Inicializando componentes...")
    llm = ChatVertexAI(
        model_name="gemini-1.5-flash",
        temperature=0.5,
        max_output_tokens=4096,
        top_p=0.95,
        top_k=40,
    )

    embeddings = VertexAIEmbeddings(model_name="text-multilingual-embedding-002")
    collection_name = "pdf-rag-chroma"

    # Inicializar vectorstore
    try:
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
            print(f"\n📄 Cargando documentos PDF desde {pdf_directory}...")
            
            # Verificar que el directorio base exista
            if not os.path.exists(pdf_directory):
                print(f"❌ El directorio {pdf_directory} no existe. Creándolo...")
                os.makedirs(pdf_directory)
                print(f"📁 Directorio creado: {pdf_directory}")
                print("❌ Coloca subcarpetas con archivos PDF y reinicia el programa.")
                return
            
            # Verificar que haya subcarpetas manualmente para debug
            print("🔍 Verificando contenido del directorio...")
            all_items = os.listdir(pdf_directory)
            subdirs = [d for d in all_items if os.path.isdir(os.path.join(pdf_directory, d))]
            files = [f for f in all_items if os.path.isfile(os.path.join(pdf_directory, f))]
            
            print(f"📊 Total elementos encontrados: {len(all_items)}")
            print(f"📂 Subdirectorios: {', '.join(subdirs) if subdirs else 'Ninguno'}")
            print(f"📄 Archivos: {', '.join(files) if files else 'Ninguno'}")
            
            if not subdirs:
                print(f"❌ No se encontraron subcarpetas en {pdf_directory}.")
                print("❌ Crear subcarpetas como 'orientaciones', 'propuesta', etc. con archivos PDF.")
                return
            
            # Cargar todos los documentos de todas las subcarpetas
            documents = load_pdf_documents(pdf_directory)
            
            if not documents:
                print("❌ No se encontraron documentos PDF en ninguna subcarpeta.")
                return

            print(f"✅ {len(documents)} páginas cargadas de {len(subdirs)} categorías")
            
            # Crear directorio para la base de datos si no existe
            os.makedirs(os.path.dirname(persist_directory), exist_ok=True)
            
            print(f"\n🔄 Creando base de datos en {persist_directory}...")
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=persist_directory
            )
            
            collection_size = len(vectorstore.get()["ids"])
            print(f"✅ Nuevo vectorstore creado con {collection_size} documentos")

    except Exception as e:
        print(f"\n❌ Error al inicializar la base de datos: {str(e)}")
        print(f"Detalles: {type(e).__name__}")
        import traceback
        traceback.print_exc()
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
    
    # Estado para mantener información entre turnos de conversación
    session_state = {
        "pending_request": False,
        "last_query": "",
        "asignatura": None,
        "nivel": None,
        "mes": None,
        "tipo": None
    }

    while True:
        try:
            query = input("\n👤 Usuario: ").strip()
            
            if query.lower() in ["exit", "quit", "q", "salir"]:
                print("\n👋 ¡Hasta luego!")
                break

            # Si hay una solicitud pendiente de información
            if session_state["pending_request"]:
                # La nueva consulta podría contener la asignatura o el nivel
                if not session_state["asignatura"]:
                    session_state["asignatura"] = query
                    print("\n🔄 Información registrada. Procesando...")
                    # Volvemos a llamar al router con la nueva información
                    response, needs_info, info, tipo = router(
                        session_state["last_query"], 
                        session_state["asignatura"], 
                        session_state["nivel"],
                        session_state["mes"]
                    )
                elif not session_state["nivel"]:
                    session_state["nivel"] = query
                    print("\n🔄 Información registrada. Procesando...")
                    # Volvemos a llamar al router con la nueva información
                    response, needs_info, info, tipo = router(
                        session_state["last_query"], 
                        session_state["asignatura"], 
                        session_state["nivel"],
                        session_state["mes"]
                    )
                
                if needs_info:
                    # Aún falta información
                    print(f"\n❓ {response}")
                    session_state["pending_request"] = True
                    # No actualizamos last_query aquí, mantenemos la consulta original
                else:
                    # Tenemos toda la información, mostrar respuesta final
                    print(f"\n🤖 Respuesta: {response}")
                    
                    # Preparar información para guardar
                    mes_info = ""
                    if info.get("mes"):
                        mes_info = f", Mes: {info.get('mes')}"
                    else:
                        # Si no hay mes específico, usamos el actual
                        meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", 
                                "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
                        mes_actual = datetime.datetime.now().month
                        mes_nombre = meses[mes_actual - 1]
                        mes_info = f", Mes: {mes_nombre} (actual)"
                    
                    # Guardar conversación usando la función correcta
                    from utils.conversation import format_and_save_conversation
                    full_query = f"{query} (Asignatura: {info.get('asignatura')}, Nivel: {info.get('nivel')}{mes_info})"
                    format_and_save_conversation(full_query, response, thread_id)
                    
                    # Aseguramos que el estado quede limpio para la próxima consulta
                    session_state = {
                        "pending_request": False,
                        "last_query": "",
                        "asignatura": None,
                        "nivel": None,
                        "mes": None,
                        "tipo": None
                    }
                    print("\n✅ El agente está listo para una nueva consulta.")
            else:
                # Nueva solicitud
                print("\n🔄 Procesando tu solicitud...")
                # Reiniciamos el estado para cada nueva solicitud
                session_state = {
                    "pending_request": False,
                    "last_query": query,
                    "asignatura": None,
                    "nivel": None,
                    "mes": None,
                    "tipo": None
                }
                # Llamamos al router sin parámetros para que analice la consulta desde cero
                response, needs_info, info, tipo = router(query)
                
                if needs_info:
                    # Si falta información, activamos el estado de solicitud pendiente
                    print(f"\n❓ {response}")
                    session_state = {
                        "pending_request": True,
                        "last_query": query,
                        "asignatura": info.get("asignatura"),
                        "nivel": info.get("nivel"),
                        "mes": info.get("mes"),
                        "tipo": tipo
                    }
                else:
                    # Si tenemos toda la información y obtuvimos respuesta
                    print(f"\n🤖 Respuesta: {response}")
                    
                    # Preparar información para guardar
                    mes_info = ""
                    if info.get("mes"):
                        mes_info = f", Mes: {info.get('mes')}"
                    else:
                        # Si no hay mes específico, usamos el actual
                        meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", 
                                "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
                        mes_actual = datetime.datetime.now().month
                        mes_nombre = meses[mes_actual - 1]
                        mes_info = f", Mes: {mes_nombre} (actual)"
                    
                    # Guardar conversación usando la función correcta
                    from utils.conversation import format_and_save_conversation
                    full_query = f"{query} (Asignatura: {info.get('asignatura')}, Nivel: {info.get('nivel')}{mes_info})"
                    format_and_save_conversation(full_query, response, thread_id)
                    
                    # Aseguramos que el estado quede limpio para la próxima consulta
                    session_state = {
                        "pending_request": False,
                        "last_query": "",
                        "asignatura": None,
                        "nivel": None,
                        "mes": None,
                        "tipo": None
                    }
                    print("\n✅ El agente está listo para una nueva consulta.")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Por favor, intenta reformular tu solicitud.")
            # Reiniciamos el estado en caso de error para empezar fresco
            session_state = {
                "pending_request": False,
                "last_query": "",
                "asignatura": None,
                "nivel": None,
                "mes": None,
                "tipo": None
            }
            print("\n✅ El agente ha sido reiniciado y está listo para una nueva consulta.")
        
if __name__ == "__main__":
    main()