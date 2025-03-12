import os
import uuid
import logging
from fastapi import FastAPI
from core.llm import get_llm
from core.vectorstore.loader import initialize_vectorstore
from core.agents.router_agent import create_router_agent

# Importar el router de chat
from api.routes.chat import router as chat_router

# Credenciales para usar VERTEX_AI
credentials_path = r"C:/Users/Dante/Desktop/rag_docente_sin_UI-1/src/config/credentials/appdocente-453515-41ddf072f3e5.json"
if not os.path.exists(credentials_path):
    raise FileNotFoundError(
        f"No se encontró el archivo de credenciales en: {credentials_path}")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Configuración de directorios
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PDF_DIRECTORY = os.path.join(BASE_DIR, "data", "raw", "pdf_docs")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "data", "processed", "vectorstores")

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
        
        # Asegurar que existan los directorios necesarios
        os.makedirs(PDF_DIRECTORY, exist_ok=True)
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        
        # Inicializar vectorstores
        vectorstores = initialize_vectorstore(
            pdf_directory=PDF_DIRECTORY,
            persist_directory=PERSIST_DIRECTORY
        )
        
        if not vectorstores:
            raise ValueError("No se pudo inicializar ninguna colección de vectorstore")
        
        # Mostrar resumen de colecciones
        print("\n📊 Colecciones disponibles:")
        for category, vs in vectorstores.items():
            collection_size = len(vs.get()['ids'])
            print(f"   - {category}: {collection_size} documentos")
        
        return llm, vectorstores, logger
        
    except Exception as e:
        logger.error(f"Error fatal al inicializar el sistema: {str(e)}")
        raise e

def main():
    """Punto de entrada principal del sistema en modo consola."""
    try:
        # Inicializar componentes básicos
        llm, vectorstores, logger = initialize_system()
        
        # Generar ID de sesión
        thread_id = str(uuid.uuid4())[:8]
        logger.info(f"🔑 ID de sesión: {thread_id}")

        # Crear router agent con vectorstores
        router_agent = create_router_agent(
            llm=llm,
            vectorstores=vectorstores,
            logger=logger,
            thread_id=thread_id
        )

        print("\n" + "=" * 50)
        print("🎯 Sistema listo para procesar solicitudes!")
        print("Puedes solicitar:")
        print("1. PLANIFICACIONES educativas")
        print("2. EVALUACIONES")
        print("3. GUÍAS de estudio")
        print("\nCategorías de documentos disponibles:")
        for categoria in vectorstores.keys():
            print(f"- {categoria}")
        print("=" * 50 + "\n")

        # Loop principal de conversación en modo consola
        while True:
            try:
                user_input = input("👤 Usuario: ").strip()
                if user_input.lower() in ['exit', 'quit', 'salir']:
                    print("\n👋 ¡Hasta luego!")
                    break
                
                response = router_agent(user_input, {})
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

# Instanciar FastAPI y agregar las rutas del chat
app = FastAPI()
app.include_router(chat_router)

if __name__ == "__main__":
    # Ejecutar en modo consola si se invoca directamente
    main()