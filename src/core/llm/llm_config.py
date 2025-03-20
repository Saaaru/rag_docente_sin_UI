import os
from dotenv import load_dotenv
from src.config.model_config import (
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TOP_P,
    LLM_TOP_K,
    EMBEDDING_MODEL_NAME
)
from dataclasses import dataclass
# Importamos lo necesario de Vertex AI
import vertexai
from vertexai.generative_models import GenerativeModel
from src.utils.rate_limiter import rate_limited_llm_call
from typing import Optional
import logging
from typing import TYPE_CHECKING, Optional

load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuración del modelo LLM usando valores de model_config.py"""
    model_name: str = os.getenv("LLM_MODEL_NAME", LLM_MODEL_NAME)
    temperature: float = float(os.getenv("LLM_TEMPERATURE", LLM_TEMPERATURE))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", LLM_MAX_TOKENS))
    top_p: float = float(os.getenv("LLM_TOP_P", LLM_TOP_P))
    top_k: int = int(os.getenv("LLM_TOP_K", LLM_TOP_K))

@dataclass
class EmbeddingConfig:
    """Configuración del modelo de embeddings usando valores de model_config.py"""
    model_name: str = os.getenv("EMBEDDING_MODEL_NAME", EMBEDDING_MODEL_NAME)

class RouterChatModel:
    def __init__(self, model_name: str = LLM_MODEL_NAME):
        """Inicializa el modelo usando la configuración por defecto de model_config.py"""
        try:
            self.config = LLMConfig()
            self.model = GenerativeModel(model_name)
            self.chat = self.model.start_chat()
            logger.info(f"Modelo {model_name} inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar el modelo: {e}")
            raise
    
    @rate_limited_llm_call
    def invoke(self, prompt: str) -> Optional[str]:
        """Invoca el modelo usando los parámetros de configuración definidos"""
        try:
            response = self.chat.send_message(
                prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "max_output_tokens": self.config.max_tokens
                }
            )
            
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].content
            else:
                logger.warning("Formato de respuesta no reconocido")
                return None
                
        except Exception as e:
            logger.error(f"Error en la generación de contenido: {e}")
            return None
    
    def reset(self) -> bool:
        """Reinicia la sesión de chat."""
        try:
            self.chat = self.model.start_chat()
            return True
        except Exception as e:
            logger.error(f"Error al reiniciar el chat: {e}")
            return False

def get_llm(model_name: str = LLM_MODEL_NAME) -> Optional[RouterChatModel]:
    """Obtiene una instancia del modelo LLM usando la configuración por defecto."""
    try:
        return RouterChatModel(model_name)
    except Exception as e:
        logger.error(f"Error al inicializar el modelo: {e}")
        return None

def get_llm_config() -> dict:
    """Obtiene la configuración actual del modelo desde model_config.py"""
    config = LLMConfig()
    return {
        "model_name": config.model_name,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_tokens": config.max_tokens
    }

def get_embedding_config() -> dict:
    """Obtiene la configuración actual de embeddings desde model_config.py"""
    config = EmbeddingConfig()
    return {
        "model_name": config.model_name,
        "dimension": 768  # Este valor podría moverse también a model_config.py
    }