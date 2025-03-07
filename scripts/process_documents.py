from src.core.vectorstore.loader import load_pdf_documents
from src.core.embeddings.embedding_utils import get_embeddings
from src.core.vectorstore.store import create_vectorstore
from src.config.paths import PDF_DIRECTORY, COLLECTION_NAME

def main():
    """Procesa los documentos PDF y crea/actualiza el vectorstore"""
    print("Procesando documentos...")
    
    # Cargar documentos
    documents = load_pdf_documents(PDF_DIRECTORY)
    if not documents:
        print("No se encontraron documentos para procesar.")
        return
    
    # Crear vectorstore
    embeddings = get_embeddings()
    vectorstore = create_vectorstore(documents, embeddings, COLLECTION_NAME)
    print("Procesamiento completado.")

if __name__ == "__main__":
    main() 