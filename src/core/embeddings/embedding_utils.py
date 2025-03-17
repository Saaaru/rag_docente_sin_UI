from langchain_google_vertexai import VertexAIEmbeddings
from core.llm.llm_config import load_credentials
from config.model_config import EMBEDDING_MODEL_NAME
import vertexai

def get_embeddings(credentials_path: str = "src/config/credentials/proyecto-docente-453715-b625fbe2c520.json"):
    """Obtiene una instancia de VertexAIEmbeddings, cargando las credenciales."""
    credentials = load_credentials(credentials_path)
    if credentials:
        try:
            # Inicializar VertexAI primero
            vertexai.init(
                project="proyecto-docente-453715",
                credentials=credentials
            )
            
            # Crear embeddings SIN el project_id
            embeddings = VertexAIEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                credentials=credentials
            )
            print("✅ Embeddings de Vertex AI inicializados correctamente.")
            return embeddings
        except Exception as e:
            print(f"❌ Error al inicializar los embeddings de Vertex AI: {e}")
            return None
    else:
        return None