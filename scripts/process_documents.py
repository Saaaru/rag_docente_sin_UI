from src.core.vectorstore.loader import initialize_vectorstore
from src.config.paths import RAW_DIR, PERSIST_DIRECTORY

def main():
    """Procesa los documentos PDF y crea/actualiza el vectorstore"""
    print("Procesando documentos...")
    
    # Inicializar el vectorstore (esto carga o crea las colecciones)
    vectorstores = initialize_vectorstore(persist_directory=str(PERSIST_DIRECTORY))

    if vectorstores:
        print("Procesamiento completado.")
    else:
        print("Error: No se pudo inicializar el vectorstore.")

if __name__ == "__main__":
    main() 