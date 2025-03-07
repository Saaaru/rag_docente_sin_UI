# Configuración del modelo LLM
LLM_MODEL_NAME = "gemini-1.5-flash"
LLM_TEMPERATURE = 0.5
LLM_MAX_TOKENS = 4096
LLM_TOP_P = 0.95
LLM_TOP_K = 40

# Configuración del modelo de embeddings
EMBEDDING_MODEL_NAME = "text-multilingual-embedding-002"

# Configuración de rate limiting
CALLS_PER_MINUTE = 150
PERIOD = 60
WAIT_TIME = 1

__all__ = [
    'LLM_MODEL_NAME',
    'LLM_TEMPERATURE',
    'LLM_MAX_TOKENS',
    'LLM_TOP_P',
    'LLM_TOP_K',
    'EMBEDDING_MODEL_NAME',
    'CALLS_PER_MINUTE',
    'PERIOD',
    'WAIT_TIME'
]