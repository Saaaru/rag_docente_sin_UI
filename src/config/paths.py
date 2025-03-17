import os
from pathlib import Path
import logging

# Configurar logging
logger = logging.getLogger(__name__)

# Obtener el directorio actual del script (paths.py)
# Este es el punto de referencia *absoluto* y confiable.
CURRENT_DIR = Path(__file__).parent.resolve()

# Asumiendo que 'data' y 'src' están *al mismo nivel* que el directorio que contiene paths.py:
#  src/
#  data/
#  (directorio donde está paths.py)
SRC_DIR = CURRENT_DIR.parent  #Directorio "padre" del actual
DATA_DIR = SRC_DIR.parent / "data" #Un nivel arriba del actual

# Directorios principales
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CREDENTIALS_DIR = CURRENT_DIR / "credentials"  # <- ¡AQUÍ ESTÁ EL CAMBIO! Ahora apunta a src/config/credentials

# Archivos de configuración y credenciales
ENV_PATH = CREDENTIALS_DIR / ".env"

# Configuración de Chroma
COLLECTION_NAME = "pdf-rag-chroma"
PERSIST_DIRECTORY = PROCESSED_DIR / COLLECTION_NAME

def verify_raw_dir() -> bool:
    """Verifica que el directorio RAW_DIR exista y tenga subdirectorios con PDFs."""
    if not RAW_DIR.exists():
        logger.error(f"❌ Error crítico: El directorio {RAW_DIR} no existe.")
        return False

    subdirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    if not subdirs:
        logger.warning(f"⚠️ Advertencia: No se encontraron subdirectorios en {RAW_DIR}")
        return False

    logger.info(f"✅ Directorio de PDFs verificado: {RAW_DIR}")
    logger.info(f"📁 Subdirectorios encontrados: {len(subdirs)}")
    return True

def verify_credentials_dir() -> bool:
    """Verifica que el directorio de credenciales exista y contenga el archivo JSON."""
    if not CREDENTIALS_DIR.exists():
        logger.error(f"❌ Error crítico: El directorio {CREDENTIALS_DIR} no existe.")
        return False
    
    # Verificar si hay archivos .json en el directorio
    json_files = list(CREDENTIALS_DIR.glob("*.json"))
    if not json_files:
        logger.error(f"❌ Error crítico: No se encontraron archivos .json en {CREDENTIALS_DIR}")
        return False

    logger.info(f"✅ Directorio de credenciales verificado: {CREDENTIALS_DIR}")
    logger.info(f"📄 Archivo de credenciales encontrado: {json_files[0].name}")
    return True

def ensure_directories():
    """
    Asegura que los directorios necesarios existan.
    Crea los directorios no críticos si no existen.
    """
    # Verificar directorio raw (crítico)
    raw_ok = verify_raw_dir()
    
    # Verificar directorio credentials (crítico para Vertex AI)
    creds_ok = verify_credentials_dir()
    
    # Crear directorios no críticos si no existen
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CONVERSATION_DIRECTORY = DATA_DIR / "conversations"
    CONVERSATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    
    return raw_ok and creds_ok

# Llamar a la función de verificación
ensure_directories()

# Definir rutas específicas
CONVERSATION_DIRECTORY = DATA_DIR / "conversations"

__all__ = [
    "RAW_DIR",
    "PERSIST_DIRECTORY",
    "COLLECTION_NAME",
    "CONVERSATION_DIRECTORY",
    "ENV_PATH",  # Añadido ENV_PATH a __all__
    "CREDENTIALS_DIR",
    "SRC_DIR",
    "ensure_directories"
]