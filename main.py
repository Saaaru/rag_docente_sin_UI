import os
import sys
import uuid
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI

# Importar módulos propios
from utils import format_and_save_conversation
from vectorstore import load_pdf_documents, create_vectorstore, create_enhanced_retriever_tool
from agents import (
    create_planning_agent,
    create_evaluation_agent,
    create_study_guide_agent,
    create_router_agent
)

def main():
    print("Inicializando Sistema Multi-Agente Educativo...")

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

    # Inicializar vectorstore
    try:
        collection_name = "pdf-rag-chroma"
        persist_directory = f"./{collection_name}"

        if os.path.exists(persist_directory):
            print("\n📚 Cargando base de datos existente...")
            vectorstore = create_vectorstore(None, embeddings, collection_name)
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
                        import datetime
                        meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", 
                                "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
                        mes_actual = datetime.datetime.now().month
                        mes_nombre = meses[mes_actual - 1]
                        mes_info = f", Mes: {mes_nombre} (actual)"
                    
                    # Guardar con información completa
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
                # Nueva solicitud - SIEMPRE EMPIEZA COMO NUEVA
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
                        import datetime
                        meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", 
                                "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
                        mes_actual = datetime.datetime.now().month
                        mes_nombre = meses[mes_actual - 1]
                        mes_info = f", Mes: {mes_nombre} (actual)"
                    
                    # Guardar con información completa
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