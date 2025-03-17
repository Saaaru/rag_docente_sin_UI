import os
from dotenv import load_dotenv
from config.model_config import (
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TOP_P,
    LLM_TOP_K
)
from dataclasses import dataclass
# Importamos lo necesario de Vertex AI
import vertexai
from vertexai.generative_models import GenerativeModel

load_dotenv()

@dataclass
class LLMConfig:
    model_name: str = os.getenv("LLM_MODEL_NAME", LLM_MODEL_NAME) # Usamos valores por defecto de model_config
    temperature: float = float(os.getenv("LLM_TEMPERATURE", LLM_TEMPERATURE))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", LLM_MAX_TOKENS))
    top_p: float = float(os.getenv("LLM_TOP_P", LLM_TOP_P))
    top_k: int = int(os.getenv("LLM_TOP_K", LLM_TOP_K))
    # project_id y location ya no son necesarios aquí, se manejan en main.py

@dataclass
class EmbeddingConfig:  # Esta clase no cambia
    model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-004")

def get_llm_config() -> LLMConfig:
    return LLMConfig()

def get_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig()

# --- Función para obtener el LLM (usando la configuración) ---
def get_llm():
    """
    Obtiene una instancia del modelo de lenguaje de Vertex AI,
    utilizando la configuración de LLMConfig.
    """
    config = get_llm_config()
    try:
        # ¡Importante!  vertexai.init() ya se llama en main.py,
        # así que aquí *no* necesitamos credenciales.
        model = GenerativeModel(config.model_name)
        return model
    except Exception as e:
        print(f"Error al obtener el LLM: {e}")
        return None