import os
import sys
from typing import List, Dict, Any
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings

# Añadir el directorio src al path para importaciones relativas
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if src_dir not in sys.path:
    sys.path.append(src_dir)

def load_pdf_documents(directory_path: str) -> List:
    """
    Carga todos los documentos PDF de un directorio y sus subdirectorios recursivamente.

    Args:
        directory_path: Ruta al directorio que contiene archivos PDF

    Returns:
        Lista de objetos Document con el contenido de los PDFs
    """
    if not os.path.exists(directory_path):
        print(f"❌ Directory {directory_path} does not exist.")
        return []

    # Verificar que la ruta sea a la carpeta pdf_docs correcta
    folder_name = os.path.basename(directory_path)
    if folder_name != "pdf_docs":
        print(f"⚠️ Aviso: El directorio no parece ser 'pdf_docs' sino '{folder_name}'")
        
    # Verificar las subcarpetas esperadas
    expected_subdirs = ["propuesta", "orientaciones", "leyes", "bases curriculares", "actividades sugeridas"]
    subdirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    
    found_subdirs = set(subdirs).intersection(set(expected_subdirs))
    print(f"📂 Subcarpetas encontradas: {', '.join(found_subdirs)}")
    
    missing_subdirs = set(expected_subdirs) - set(subdirs)
    if missing_subdirs:
        print(f"⚠️ Subcarpetas no encontradas: {', '.join(missing_subdirs)}")

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
                
                # Añadir metadatos de categoría basados en la subcarpeta
                for doc in docs:
                    # Obtener la primera parte de la ruta relativa como categoría
                    rel_path = os.path.relpath(root, directory_path)
                    category = rel_path.split(os.sep)[0] if os.sep in rel_path else rel_path
                    doc.metadata["category"] = category
                
                documents.extend(docs)
                processed_files += 1
                print(f"    ✅ {len(docs)} páginas cargadas")
            except Exception as e:
                print(f"    ❌ Error al cargar {pdf_file}: {e}")

    print(f"\n📊 Resumen:")
    print(f"   - {processed_files}/{total_files} archivos procesados")
    print(f"   - {len(documents)} páginas totales cargadas")

    return documents

def process_documents(documents: List, chunk_size: int = 1500, chunk_overlap: int = 300) -> List:
    """
    Procesa documentos dividiéndolos en chunks para mejor manejo.
    
    Args:
        documents: Lista de documentos a procesar
        chunk_size: Tamaño de cada chunk
        chunk_overlap: Superposición entre chunks
        
    Returns:
        Lista de chunks procesados
    """
    if not documents:
        print("❌ No hay documentos para procesar")
        return []
    
    print(f"\n✂️ Dividiendo documentos en chunks (tamaño={chunk_size}, superposición={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ {len(documents)} documentos divididos en {len(chunks)} chunks")
    
    return chunks

def create_vectorstore(documents, persist_directory=None, collection_name="pdf-rag-chroma"):
    """
    Crea un Chroma vector store a partir de documentos.
    Primero divide los documentos en chunks y luego los almacena en la base de datos.

    Args:
        documents: Lista de documentos a procesar
        persist_directory: Directorio donde guardar la base de datos (opcional)
        collection_name: Nombre de la colección

    Returns:
        El vectorstore de Chroma creado
    """
    # Inicializar embeddings de VertexAI
    embeddings = VertexAIEmbeddings(model_name="text-multilingual-embedding-002")
    print("✅ Embeddings inicializados")

    # Dividir documentos en chunks
    print("📄 Procesando documentos...")
    chunks = process_documents(documents)

    # Crear vectorstore
    print("🔄 Creando vectorstore...")
    if persist_directory:
        # Asegurarse de que el directorio exista
        os.makedirs(persist_directory, exist_ok=True)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        print(f"✅ Vectorstore creado y guardado en {persist_directory}")
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name
        )
        print("✅ Vectorstore creado en memoria")

    return vectorstore

def initialize_vectorstore(pdf_directory: str, persist_directory: str, collection_name: str = "pdf-rag-chroma") -> Chroma:
    """
    Inicializa la vectorstore: carga una existente o crea una nueva si no existe.
    
    Args:
        pdf_directory: Directorio donde están los PDFs
        persist_directory: Directorio donde guardar/cargar la vectorstore
        collection_name: Nombre de la colección
        
    Returns:
        Chroma: La vectorstore inicializada
    """
    print("📚 Inicializando vectorstore...")
    
    # Verificar si existe la base de datos
    if os.path.exists(persist_directory):
        print("   Cargando base de datos existente...")
        try:
            # Inicializar embeddings
            embeddings = VertexAIEmbeddings(model_name="text-multilingual-embedding-002")
            
            # Cargar vectorstore existente
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            
            # Verificar que tenga documentos
            collection_size = len(vectorstore.get()['ids'])
            if collection_size == 0:
                raise ValueError("La base de datos existe pero está vacía")
                
            print(f"✅ Base de datos cargada con {collection_size} documentos")
            return vectorstore
            
        except Exception as e:
            print(f"⚠️ Error al cargar base de datos existente: {e}")
            print("   Intentando crear nueva base de datos...")
    
    # Si no existe o hubo error, crear nueva
    print("   Creando nueva base de datos...")
    
    # Cargar documentos
    documents = load_pdf_documents(pdf_directory)
    if not documents:
        raise ValueError("No se encontraron documentos PDF para procesar")
    
    # Crear vectorstore
    vectorstore = create_vectorstore(
        documents=documents,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    return vectorstore

def main():
    """Función principal para cargar PDFs y crear vectorstore."""
    # Determinar rutas
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pdf_directory = os.path.join(base_dir, "data", "raw", "pdf_docs")
    persist_directory = os.path.join(base_dir, "data", "processed", "pdf-rag-chroma")
    
    # Si estamos en Windows, manejar ruta absoluta
    if os.name == 'nt' and not os.path.exists(pdf_directory):
        # Intenta usar la ruta absoluta conocida
        pdf_directory = r"C:\Users\mfuen\OneDrive\Desktop\rag_docente_sin_UI\data\raw\pdf_docs"
    
    print(f"🔍 Directorio de PDFs: {pdf_directory}")
    print(f"💾 Directorio para persistencia: {persist_directory}")

    # Verificar que el directorio de PDFs existe
    if not os.path.exists(pdf_directory):
        print(f"❌ El directorio de PDFs no existe: {pdf_directory}")
        return
    
    # Crear directorio de persistencia si no existe
    os.makedirs(os.path.dirname(persist_directory), exist_ok=True)
    
    # Crear vectorstore
    vectorstore = initialize_vectorstore(
        pdf_directory=pdf_directory,
        persist_directory=persist_directory,
        collection_name="pdf-rag-chroma"
    )
    
    # Verificar que se haya creado correctamente
    collection_size = len(vectorstore.get()['ids'])
    print(f"✅ Vectorstore creado con {collection_size} documentos")

if __name__ == "__main__":
    main()
