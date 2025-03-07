from langchain_community.document_loaders import PyPDFLoader
import os
from typing import List 

def load_pdf_documents(directory_path: str) -> List:
    """
    Loads all PDF documents from a given directory and its subdirectories recursively.

    Args:
        directory_path: Path to the directory containing PDF files

    Returns:
        List of Document objects containing the content of the PDFs
    """
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return []

    documents = []
    total_files = 0
    processed_files = 0

    # Primero, contar total de archivos PDF
    for root, _, files in os.walk(directory_path):
        total_files += len([f for f in files if f.endswith('.pdf')])

    print(f"\n📁 Encontrados {total_files} archivos PDF en total")

    # Procesar archivos
    for root, _, files in os.walk(directory_path):
        pdf_files = [f for f in files if f.endswith('.pdf')]
        
        if pdf_files:
            # Mostrar la subcarpeta actual
            relative_path = os.path.relpath(root, directory_path)
            if relative_path != ".":
                print(f"\n📂 Procesando carpeta: {relative_path}")

        for pdf_file in pdf_files:
            try:
                file_path = os.path.join(root, pdf_file)
                print(f"  📄 Cargando: {pdf_file}")
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                processed_files += 1
                print(f"    ✅ {len(docs)} páginas cargadas")
            except Exception as e:
                print(f"    ❌ Error al cargar {pdf_file}: {e}")

    print(f"\n📊 Resumen:")
    print(f"   - {processed_files}/{total_files} archivos procesados")
    print(f"   - {len(documents)} páginas totales cargadas")

    return documents
