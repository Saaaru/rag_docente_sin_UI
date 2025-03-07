from langchain_google_vertexai import ChatVertexAI
from config.model_config import (
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TOP_P,
    LLM_TOP_K
)
from dataclasses import dataclass

@dataclass
class LLMConfig:
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.5
    max_tokens: int = 4096
    top_p: float = 0.95
    top_k: int = 40
    project_id: str = "gen-lang-client-0115469242"
    location: str = "us-central1"

@dataclass
class EmbeddingConfig:
    model_name: str = "text-multilingual-embedding-002"

def get_llm():
    """Inicializa y retorna el modelo LLM configurado"""
    return ChatVertexAI(
        model_name=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
        top_p=LLM_TOP_P,
        top_k=LLM_TOP_K,
    )

def get_llm_config() -> LLMConfig:
    return LLMConfig()

def get_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig()