import os
import sys
from typing import List, Dict, Any, Optional
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

def load_pdf_documents(directory_path: str) -> Dict[str, List]:
    """
    Carga documentos PDF organizados por subcarpetas.

    Args:
        directory_path: Ruta al directorio que contiene las subcarpetas con PDFs

    Returns:
        Dict con subcarpetas como claves y listas de documentos como valores
    """
    if not os.path.exists(directory_path):
        print(f"❌ Directory {directory_path} does not exist.")
        return {}

    # Verificar que la ruta sea a la carpeta pdf_docs correcta
    folder_name = os.path.basename(directory_path)
    if folder_name != "pdf_docs":
        print(f"⚠️ Aviso: El directorio no parece ser 'pdf_docs' sino '{folder_name}'")
    
    # Subcarpetas esperadas
    expected_subdirs = ["propuesta", "orientaciones", "leyes", "bases curriculares", "actividades sugeridas"]
    documents_by_category = {subdir: [] for subdir in expected_subdirs}
    
    # Verificar subcarpetas existentes
    subdirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    found_subdirs = set(subdirs).intersection(set(expected_subdirs))
    print(f"📂 Subcarpetas encontradas: {', '.join(found_subdirs)}")
    
    missing_subdirs = set(expected_subdirs) - set(subdirs)
    if missing_subdirs:
        print(f"⚠️ Subcarpetas no encontradas: {', '.join(missing_subdirs)}")

    total_files = 0
    processed_files = 0

    # Contar total de archivos PDF por categoría
    for category in found_subdirs:
        category_path = os.path.join(directory_path, category)
        for root, _, files in os.walk(category_path):
            total_files += len([f for f in files if f.endswith('.pdf')])

    print(f"\n📁 Encontrados {total_files} archivos PDF en total")

    # Procesar archivos por categoría
    for category in found_subdirs:
        category_path = os.path.join(directory_path, category)
        print(f"\n📂 Procesando categoría: {category}")
        
        for root, _, files in os.walk(category_path):
            pdf_files = [f for f in files if f.endswith('.pdf')]
            
            for pdf_file in pdf_files:
                try:
                    file_path = os.path.join(root, pdf_file)
                    print(f"  📄 Cargando: {pdf_file}")
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    
                    # Añadir metadatos de categoría
                    for doc in docs:
                        doc.metadata["category"] = category
                    
                    documents_by_category[category].extend(docs)
                    processed_files += 1
                    print(f"    ✅ {len(docs)} páginas cargadas")
                except Exception as e:
                    print(f"    ❌ Error al cargar {pdf_file}: {e}")

    # Resumen final
    print(f"\n📊 Resumen por categoría:")
    for category, docs in documents_by_category.items():
        if docs:
            print(f"   - {category}: {len(docs)} páginas")
    print(f"\n📊 Total:")
    print(f"   - {processed_files}/{total_files} archivos procesados")
    total_pages = sum(len(docs) for docs in documents_by_category.values())
    print(f"   - {total_pages} páginas totales")

    return documents_by_category

def process_documents(documents: Dict[str, List], chunk_size: int = 1500, chunk_overlap: int = 300) -> Dict[str, List]:
    """
    Procesa documentos por categoría dividiéndolos en chunks.
    
    Args:
        documents: Diccionario de documentos por categoría
        chunk_size: Tamaño de cada chunk
        chunk_overlap: Superposición entre chunks
        
    Returns:
        Diccionario de chunks procesados por categoría
    """
    if not documents:
        print("❌ No hay documentos para procesar")
        return {}
    
    processed_documents = {}
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    
    print(f"\n✂️ Dividiendo documentos en chunks (tamaño={chunk_size}, superposición={chunk_overlap})...")
    
    for category, docs in documents.items():
        if docs:
            print(f"   📑 Procesando {category}...")
            chunks = text_splitter.split_documents(docs)
            processed_documents[category] = chunks
            print(f"      ✅ {len(docs)} documentos divididos en {len(chunks)} chunks")
    
    return processed_documents

def create_vectorstore(documents: Dict[str, List], persist_directory: str) -> Dict[str, Chroma]:
    """
    Crea colecciones de Chroma para cada categoría de documentos.

    Args:
        documents: Diccionario de documentos por categoría
        persist_directory: Directorio base donde guardar las colecciones

    Returns:
        Diccionario de vectorstores por categoría
    """
    # Asegurar que exista el directorio base
    os.makedirs(persist_directory, exist_ok=True)
    
    # Inicializar embeddings de VertexAI (se comparte entre todas las colecciones)
    embeddings = VertexAIEmbeddings(model_name="text-multilingual-embedding-002")
    print("✅ Embeddings inicializados")

    vectorstores = {}
    
    for category, docs in documents.items():
        if not docs:
            continue
            
        print(f"\n📑 Procesando categoría: {category}")
        
        # Crear directorio específico para la categoría
        category_dir = os.path.join(persist_directory, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Definir archivos de la colección
        collection_name = f"pdf-rag-{category}"
        metadata_file = os.path.join(category_dir, "metadata.json")
        
        try:
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=category_dir
            )
            vectorstores[category] = vectorstore
            print(f"✅ Vectorstore creado para {category} con {len(docs)} documentos")
            print(f"   💾 Guardado en {category_dir}")
            
            # Guardar metadata de la colección
            from datetime import datetime
            metadata = {
                "category": category,
                "document_count": len(docs),
                "created_at": datetime.now().isoformat(),
                "collection_name": collection_name,
                "directory": category_dir
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"❌ Error al crear vectorstore para {category}: {e}")
    
    return vectorstores

def initialize_vectorstore(pdf_directory: str, persist_directory: str) -> Dict[str, Chroma]:
    """
    Inicializa las colecciones de vectorstore: carga existentes o crea nuevas.
    
    Args:
        pdf_directory: Directorio donde están los PDFs organizados por subcarpetas
        persist_directory: Directorio base donde guardar/cargar las colecciones
        
    Returns:
        Dict[str, Chroma]: Diccionario de vectorstores por categoría
    """
    print("📚 Inicializando vectorstores...")
    
    # Asegurar que existan los directorios necesarios
    os.makedirs(persist_directory, exist_ok=True)
    
    # Inicializar embeddings (se compartirá entre todas las colecciones)
    embeddings = VertexAIEmbeddings(model_name="text-multilingual-embedding-002")
    
    vectorstores = {}
    expected_categories = ["propuesta", "orientaciones", "leyes", "bases curriculares", "actividades sugeridas"]
    
    # Crear estructura de directorios para vectorstores
    for category in expected_categories:
        category_dir = os.path.join(persist_directory, category)
        os.makedirs(category_dir, exist_ok=True)
    
    # Intentar cargar colecciones existentes
    for category in expected_categories:
        category_dir = os.path.join(persist_directory, category)
        collection_name = f"pdf-rag-{category}"
        metadata_file = os.path.join(category_dir, "metadata.json")
        
        if os.path.exists(category_dir):
            print(f"\n📂 Intentando cargar colección existente para {category}...")
            try:
                vectorstore = Chroma(
                    persist_directory=category_dir,
                    embedding_function=embeddings,
                    collection_name=collection_name
                )
                
                # Verificar que tenga documentos
                collection_size = len(vectorstore.get()['ids'])
                if collection_size == 0:
                    raise ValueError(f"La colección {category} existe pero está vacía")
                    
                vectorstores[category] = vectorstore
                print(f"✅ Colección {category} cargada con {collection_size} documentos")
                
                # Guardar o actualizar metadata
                from datetime import datetime
                metadata = {
                    "category": category,
                    "document_count": collection_size,
                    "last_loaded": datetime.now().isoformat(),
                    "collection_name": collection_name,
                    "directory": category_dir
                }
                
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                print(f"⚠️ Error al cargar colección {category}: {e}")
                print(f"   Se creará una nueva colección para {category}")
    
    # Si faltan colecciones, crear nuevas
    if len(vectorstores) < len(expected_categories):
        print("\n📄 Cargando documentos para colecciones faltantes...")
        
        # Cargar documentos
        documents = load_pdf_documents(pdf_directory)
        if not documents:
            raise ValueError("No se encontraron documentos PDF para procesar")
        
        # Procesar documentos
        processed_docs = process_documents(documents)
        
        # Crear vectorstores faltantes
        for category in expected_categories:
            if category not in vectorstores and category in processed_docs:
                category_dir = os.path.join(persist_directory, category)
                collection_name = f"pdf-rag-{category}"
                metadata_file = os.path.join(category_dir, "metadata.json")
                
                try:
                    vectorstore = Chroma.from_documents(
                        documents=processed_docs[category],
                        embedding=embeddings,
                        collection_name=collection_name,
                        persist_directory=category_dir
                    )
                    vectorstores[category] = vectorstore
                    print(f"✅ Nueva colección creada para {category}")
                    
                    # Guardar metadata de la nueva colección
                    from datetime import datetime
                    metadata = {
                        "category": category,
                        "document_count": len(processed_docs[category]),
                        "created_at": datetime.now().isoformat(),
                        "collection_name": collection_name,
                        "directory": category_dir
                    }
                    
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        import json
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    
                except Exception as e:
                    print(f"❌ Error al crear colección {category}: {e}")
    
    return vectorstores

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
    vectorstores = initialize_vectorstore(
        pdf_directory=pdf_directory,
        persist_directory=persist_directory
    )
    
    # Verificar que se hayan creado correctamente
    for category, vectorstore in vectorstores.items():
        collection_size = len(vectorstore.get()['ids'])
        print(f"✅ Vectorstore creado para {category} con {collection_size} documentos")

if __name__ == "__main__":
    main()
