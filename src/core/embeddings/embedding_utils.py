from langchain_google_vertexai import VertexAIEmbeddings
from config.model_config import EMBEDDING_MODEL_NAME

def get_embeddings():
    """Inicializa y retorna el modelo de embeddings"""
    return VertexAIEmbeddings(model_name=EMBEDDING_MODEL_NAME)