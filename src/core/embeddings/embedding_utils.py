from langchain_google_vertexai import VertexAIEmbeddings
from config.model_config import EMBEDDING_MODEL_NAME
import os
import logging

logger = logging.getLogger(__name__)

def get_embeddings():
    """Obtiene una instancia de VertexAIEmbeddings."""
    try:
        # Usar el mismo PROJECT_ID que se usa en main.py
        project_id = os.environ.get("GOOGLE_PROJECT_ID")
        if not project_id:
            raise ValueError("GOOGLE_PROJECT_ID no está configurado")

        embeddings = VertexAIEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            project=project_id,  # Especificar el project_id explícitamente
        )
        logger.info("✅ Embeddings de Vertex AI inicializados correctamente.")
        return embeddings
    except Exception as e:
        logger.error(f"❌ Error al inicializar los embeddings de Vertex AI: {e}")
        return None