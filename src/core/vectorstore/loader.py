import os
import sys
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
import json
from datetime import datetime

# Añadir el directorio src al path para importaciones relativas
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Añadir después de las importaciones
COLLECTION_NAMES = {
    "propuesta": "pdf-rag-propuesta",
    "orientaciones": "pdf-rag-orientaciones",
    "leyes": "pdf-rag-leyes",
    "bases curriculares": "pdf-rag-bases-curriculares",
    "actividades sugeridas": "pdf-rag-actividades-sugeridas"
}

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

def process_documents(documents: Dict[str, List], chunk_size: int = 1000, chunk_overlap: int = 150) -> Dict[str, List]:
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
    """Crea colecciones de Chroma para cada categoría de documentos."""
    os.makedirs(persist_directory, exist_ok=True)
    embeddings = VertexAIEmbeddings(model_name="text-multilingual-embedding-002")
    print("✅ Embeddings inicializados")

    vectorstores = {}
    BATCH_SIZE = 1000  # Procesar en lotes de 1000 documentos
    
    for category, docs in documents.items():
        if not docs:
            continue
            
        print(f"\n📑 Procesando categoría: {category}")
        
        collection_name = COLLECTION_NAMES.get(category)
        if not collection_name:
            print(f"⚠️ Categoría no reconocida: {category}")
            continue

        category_dir = os.path.join(persist_directory, category)
        checkpoint_file = os.path.join(category_dir, "checkpoint.json")
        os.makedirs(category_dir, exist_ok=True)
        
        # Verificar punto de control existente
        processed_count = 0
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                processed_count = checkpoint.get('processed_count', 0)
                print(f"📍 Recuperando desde checkpoint: {processed_count} documentos procesados")
        
        try:
            # Crear o cargar vectorstore existente
            if processed_count == 0:
                vectorstore = Chroma(
                    persist_directory=category_dir,
                    embedding_function=embeddings,
                    collection_name=collection_name
                )
            else:
                vectorstore = Chroma(
                    persist_directory=category_dir,
                    embedding_function=embeddings,
                    collection_name=collection_name
                )
            
            # Procesar documentos restantes en lotes
            remaining_docs = docs[processed_count:]
            total_remaining = len(remaining_docs)
            
            if total_remaining > 0:
                print(f"📦 Procesando {total_remaining} documentos restantes en lotes de {BATCH_SIZE}")
                
                for i in range(0, total_remaining, BATCH_SIZE):
                    batch = remaining_docs[i:i + BATCH_SIZE]
                    try:
                        vectorstore.add_documents(batch)
                        processed_count += len(batch)
                        
                        # Guardar checkpoint
                        with open(checkpoint_file, 'w') as f:
                            json.dump({
                                'processed_count': processed_count,
                                'last_update': datetime.now().isoformat()
                            }, f)
                        
                        print(f"✅ Procesados {processed_count}/{len(docs)} documentos")
                        
                    except Exception as batch_error:
                        print(f"⚠️ Error en lote {i//BATCH_SIZE + 1}: {batch_error}")
                        continue
            
            vectorstores[category] = vectorstore
            print(f"✅ Vectorstore completado para {category}")
            
        except Exception as e:
            print(f"❌ Error al crear colección {category}: {e}")
    
    return vectorstores

def validate_existing_collections(persist_directory: str, embeddings) -> Dict[str, Chroma]:
    """
    Valida las colecciones existentes y retorna un diccionario con las que están correctamente cargadas.
    """
    vectorstores = {}
    print("\n🔍 Validando colecciones existentes...")
    
    for category, collection_name in COLLECTION_NAMES.items():
        category_dir = os.path.join(persist_directory, category)
        if os.path.exists(category_dir):
            try:
                vectorstore = Chroma(
                    persist_directory=category_dir,
                    embedding_function=embeddings,
                    collection_name=collection_name
                )
                
                # Verificar que tenga documentos
                collection_size = len(vectorstore.get()['ids'])
                if collection_size > 0:
                    vectorstores[category] = vectorstore
                    print(f"✅ Colección {category} validada: {collection_size} documentos")
                else:
                    print(f"⚠️ Colección {category} existe pero está vacía")
            except Exception as e:
                print(f"⚠️ Error al validar colección {category}: {e}")
    
    return vectorstores

def initialize_vectorstore(pdf_directory: str, persist_directory: str) -> Dict[str, Chroma]:
    """Inicializa las colecciones de vectorstore."""
    print("📚 Inicializando vectorstores...")
    
    os.makedirs(persist_directory, exist_ok=True)
    embeddings = VertexAIEmbeddings(model_name="text-multilingual-embedding-002")
    
    # Validar colecciones existentes
    vectorstores = validate_existing_collections(persist_directory, embeddings)
    
    # Identificar categorías faltantes
    missing_categories = set(COLLECTION_NAMES.keys()) - set(vectorstores.keys())
    
    if not missing_categories:
        print("\n✨ Todas las colecciones están correctamente cargadas")
        return vectorstores
    
    print(f"\n📝 Categorías pendientes de procesar: {', '.join(missing_categories)}")
    
    # Cargar y procesar documentos solo para las categorías faltantes
    print("\n📄 Cargando documentos para colecciones faltantes...")
    documents = load_pdf_documents(pdf_directory)
    if not documents:
        raise ValueError("No se encontraron documentos PDF para procesar")
    
    # Filtrar solo los documentos de categorías faltantes
    documents_to_process = {k: v for k, v in documents.items() if k in missing_categories}
    
    # Procesar documentos
    processed_docs = process_documents(documents_to_process)
    
    # Crear vectorstores faltantes
    for category in missing_categories:
        if category in processed_docs:
            category_dir = os.path.join(persist_directory, category)
            collection_name = COLLECTION_NAMES[category]
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
                
                # Guardar metadata
                metadata = {
                    "category": category,
                    "document_count": len(processed_docs[category]),
                    "created_at": datetime.now().isoformat(),
                    "collection_name": collection_name,
                    "directory": category_dir
                }
                
                with open(metadata_file, 'w', encoding='utf-8') as f:
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
