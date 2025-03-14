from langchain_google_vertexai import VertexAIEmbeddings
from core.llm.llm_config import load_credentials, EmbeddingConfig  # Importar load_credentials

def get_embeddings(credentials_path: str = "src/config/proyecto-docente-453715-a70af6ec3ae6.json"):
    """Obtiene una instancia de VertexAIEmbeddings, cargando las credenciales."""
    credentials = load_credentials(credentials_path)
    if credentials:
        try:
            embeddings = VertexAIEmbeddings(
                model_name=EmbeddingConfig.model_name,
                credentials=credentials  # Pasar las credenciales
            )
            print("✅ Embeddings de Vertex AI inicializados correctamente.")
            return embeddings
        except Exception as e:
            print(f"❌ Error al inicializar los embeddings de Vertex AI: {e}")
            return None
    else:
        return None