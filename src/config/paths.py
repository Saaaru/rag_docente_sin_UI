import os

# Obtener el directorio base del proyecto (2 niveles arriba de este archivo)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Directorios principales
PDF_DIRECTORY = os.path.join(BASE_DIR, "data", "raw", "pdf_docs")
CREDENTIALS_DIR = os.path.join(BASE_DIR, "src", "config", "credentials")

# Archivos de configuración y credenciales
ENV_PATH = os.path.join(CREDENTIALS_DIR, ".env")

# Configuración de Chroma
COLLECTION_NAME = "pdf-rag-chroma"
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "data", "processed", COLLECTION_NAME)

# Asegurar que los directorios existan
def ensure_directories():
    """Crear los directorios necesarios si no existen"""
    directories = [
        PDF_DIRECTORY,
        CREDENTIALS_DIR,
        os.path.dirname(PERSIST_DIRECTORY)
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Directorio creado: {directory}")

# Llamar a la función cuando se importa el módulo
ensure_directories()

# Definir rutas específicas
CONVERSATION_DIRECTORY = os.path.join(BASE_DIR, "data", "conversations")

__all__ = [
    "PDF_DIRECTORY",
    "PERSIST_DIRECTORY",
    "COLLECTION_NAME",
    "CONVERSATION_DIRECTORY"
] 