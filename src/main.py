import os
import uuid
import logging
from typing import List
from core.llm import get_llm
from core.embeddings import get_embeddings
from core.vectorstore.loader import initialize_vectorstore
from core.agents import (
    create_planning_agent,
    create_evaluation_agent,
    create_study_guide_agent,
    create_router_agent
)
from config import COLLECTION_NAME

# Credenciales para usar VERTEX_AI
credentials_path = r"C:/Users/Dante/Desktop/rag_docente_sin_UI-1/src/config/credentials/gen-lang-client-0115469242-239dc466873d.json"
if not os.path.exists(credentials_path):
    raise FileNotFoundError(
        f"No se encontró el archivo de credenciales en: {credentials_path}")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Configuración de directorios y nombres de colección
COLLECTION_NAME = "pdf-rag-chroma"
PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", COLLECTION_NAME)
PDF_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "pdf_docs")
def initialize_system():
    """Inicializa los componentes básicos del sistema."""
    print("\nInicializando Sistema Multi-Agente Educativo...")
    print("⚙️ Inicializando componentes...")
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Inicializar LLM
        llm = get_llm()
        print("✅ LLM inicializado")
        
        # Inicializar embeddings
        embeddings = get_embeddings()
        print("✅ Embeddings inicializados")
        
        # Inicializar vectorstore
        vectorstore = initialize_vectorstore(
            pdf_directory=PDF_DIRECTORY,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=COLLECTION_NAME
        )
        
        return llm, vectorstore, logger
        
    except Exception as e:
        print(f"\n❌ Error fatal al inicializar el sistema: {str(e)}")
        raise e

def main():
    """Punto de entrada principal del sistema."""
    try:
        # Inicializar componentes básicos
        llm, vectorstore, logger = initialize_system()
        
        # Generar ID de sesión
        thread_id = str(uuid.uuid4())[:8]
        logger.info(f"🔑 ID de sesión: {thread_id}")

        # Crear agentes especializados
        planning_agent = create_planning_agent(llm, vectorstore)
        evaluation_agent = create_evaluation_agent(llm, vectorstore)
        study_guide_agent = create_study_guide_agent(llm, vectorstore)

        # Crear router agent
        router = create_router_agent(
            llm=llm,
            planning_agent=planning_agent,
            evaluation_agent=evaluation_agent,
            study_guide_agent=study_guide_agent,
            logger=logger,
            thread_id=thread_id
        )

        # Estado inicial de la sesión
        session_state = {
            "pending_request": False,
            "last_query": "",
            "asignatura": None,
            "nivel": None,
            "mes": None,
            "tipo": None
        }

        print("\n" + "=" * 50)
        print("🎯 Sistema listo para procesar solicitudes!")
        print("Puedes solicitar:")
        print("1. PLANIFICACIONES educativas")
        print("2. EVALUACIONES")
        print("3. GUÍAS de estudio")
        print("=" * 50 + "\n")

        # Loop principal de conversación
        while True:
            try:
                user_input = input("👤 Usuario: ").strip()
                if user_input.lower() in ['exit', 'quit', 'salir']:
                    print("\n👋 ¡Hasta luego!")
                    break
                
                response = router(user_input, session_state)
                print(f"\n🤖 Asistente: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                logger.error(f"Error en el loop principal: {e}")
                print("\n❌ Ocurrió un error. Por favor, intenta de nuevo.")

    except Exception as e:
        print(f"\n❌ Error fatal: {str(e)}")
        return

if __name__ == "__main__":
    main()