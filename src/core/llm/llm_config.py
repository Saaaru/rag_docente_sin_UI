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
from langchain_google_vertexai import ChatVertexAI  # Importar ChatVertexAI
import json
from google.oauth2 import service_account

load_dotenv()

@dataclass
class LLMConfig:
    model_name: str = os.getenv("LLM_MODEL_NAME", "gemini-1.5-pro")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", 0.5))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", 4096))
    top_p: float = float(os.getenv("LLM_TOP_P", 0.95))
    top_k: int = int(os.getenv("LLM_TOP_K", 40))
    project_id: str = "gen-lang-client-0115469242"
    location: str = "us-central1"

@dataclass
class EmbeddingConfig:
    model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-004")

def load_credentials(credentials_path: str):
    """Carga las credenciales desde el archivo JSON."""
    try:
        abs_path = os.path.abspath(credentials_path)
        if not os.path.exists(abs_path):
            print(f"❌ Error: Archivo de credenciales no encontrado en {abs_path}")
            return None
            
        credentials = service_account.Credentials.from_service_account_file(abs_path)
        print("✅ Credenciales cargadas correctamente.")
        return credentials
    except Exception as e:
        print(f"❌ Error al cargar credenciales: {e}")
        return None

def get_llm(credentials_path: str = "src/config/proyecto-docente-453715-a70af6ec3ae6.json"):
    """Obtiene una instancia del LLM de Vertex AI, cargando las credenciales."""
    credentials = load_credentials(credentials_path)
    if credentials:
        try:
            llm = ChatVertexAI(
                model_name=LLMConfig.model_name,
                temperature=LLMConfig.temperature,
                max_output_tokens=LLMConfig.max_tokens,
                top_p=LLMConfig.top_p,
                top_k=LLMConfig.top_k,
                credentials=credentials  # Pasar las credenciales directamente
            )
            print("✅ LLM de Vertex AI inicializado correctamente.")
            return llm
        except Exception as e:
            print(f"❌ Error al inicializar el LLM de Vertex AI: {e}")
            return None
    else:
        return None

def get_llm_config() -> LLMConfig:
    return LLMConfig()

def get_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig()