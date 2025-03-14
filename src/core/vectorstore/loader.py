import os
import sys
import hashlib
import json
from typing import Dict, List, Optional
from datetime import datetime
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

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
    "bases_curriculares": "pdf-rag-bases-curriculares",
    "actividades_sugeridas": "pdf-rag-actividades-sugeridas"
}

class EmbeddingsManager:
    """Singleton para gestionar una única instancia de embeddings."""
    _instance = None

    @classmethod
    def get_embeddings(cls):
        """Obtiene la instancia única de embeddings."""
        if cls._instance is None:
            cls._instance = VertexAIEmbeddings()
        return cls._instance

def _calculate_pdf_hash(pdf_path: str) -> str:
    """Calcula el hash SHA256 de un archivo PDF."""
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as pdf_file:
        while True:
            chunk = pdf_file.read(4096)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def split_large_document(doc: Document, max_tokens: int = 15000) -> List[Document]:
    """
    Divide un documento grande en secciones más manejables preservando el contexto.
    
    Args:
        doc: Documento a dividir
        max_tokens: Máximo de tokens por sección (estimado por caracteres)
    """
    max_chars = max_tokens * 4
    content = doc.page_content
    metadata = doc.metadata.copy()
    
    sections = []
    current_section = []
    current_length = 0
    
    paragraphs = content.split('\n\n')
    
    for i, paragraph in enumerate(paragraphs):
        paragraph_length = len(paragraph)
        
        if current_length + paragraph_length > max_chars and current_section:
            section_content = '\n\n'.join(current_section)
            section_metadata = metadata.copy()
            section_metadata['section'] = len(sections) + 1
            section_metadata['total_sections'] = -1
            sections.append(Document(page_content=section_content, metadata=section_metadata))
            
            current_section = []
            current_length = 0
        
        current_section.append(paragraph)
        current_length += paragraph_length
    
    if current_section:
        section_content = '\n\n'.join(current_section)
        section_metadata = metadata.copy()
        section_metadata['section'] = len(sections) + 1
        section_metadata['total_sections'] = -1
        sections.append(Document(page_content=section_content, metadata=section_metadata))
    
    total_sections = len(sections)
    for section in sections:
        section.metadata['total_sections'] = total_sections
    
    return sections

def load_category_documents(directory_path: str, category: str, batch_size: int = 100) -> List[Document]:
    """
    Carga y preprocesa documentos de una categoría específica.
    """
    category_path = os.path.join(directory_path, category)
    if not os.path.exists(category_path):
        print(f"❌ Directorio no encontrado para categoría {category}")
        return []
    
    documents = []
    total_files = len([f for f in os.listdir(category_path) if f.endswith('.pdf')])
    print(f"\n📂 Procesando {total_files} archivos de {category}")
    
    for batch_start in range(0, total_files, batch_size):
        batch_files = os.listdir(category_path)[batch_start:batch_start + batch_size]
        batch_docs: List[Document] = []
        
        for pdf_file in batch_files:
            if pdf_file.endswith('.pdf'):
                try:
                    file_path = os.path.join(category_path, pdf_file)
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    
                    processed_docs = []
                    for doc in docs:
                        estimated_tokens = len(doc.page_content) / 4
                        
                        if estimated_tokens > 15000:
                            print(f"  📑 Dividiendo documento grande: {pdf_file}")
                            sections = split_large_document(doc)
                            processed_docs.extend(sections)
                        else:
                            doc.metadata["category"] = category
                            processed_docs.append(doc)
                    
                    batch_docs.extend(processed_docs)
                    print(f"  ✅ Cargado: {pdf_file} ({len(processed_docs)} secciones)")
                    
                except Exception as e:
                    print(f"  ❌ Error en {pdf_file}: {e}")
        
        documents.extend(batch_docs)
        print(f"  📦 Lote procesado: {len(batch_docs)} documentos")
    
    return documents

def process_documents_in_batches(documents: List[Document], chunk_size: Optional[int] = None,
                               chunk_overlap: Optional[int] = None) -> List[Document]:
    """
    Procesa documentos manteniendo la integridad del contenido. Usa valores de .env si no se especifican.
    """
    chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "200"))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    
    processed_chunks = []
    
    for doc in documents:
        try:
            chunks = text_splitter.split_documents([doc])
            for chunk in chunks:
                if 'section' in doc.metadata:
                    chunk.metadata['section'] = doc.metadata['section']
                    chunk.metadata['total_sections'] = doc.metadata['total_sections']
            processed_chunks.extend(chunks)
        except Exception as e:
            print(f"  ⚠️ Error procesando documento: {e}")
            continue
    
    print(f"  ✂️ Total chunks generados: {len(processed_chunks)}")
    return processed_chunks

def create_category_vectorstore(chunks: List[Document], category: str, embeddings: VertexAIEmbeddings,
                              persist_directory: str, batch_size: int = 1000) -> Optional[Chroma]:
    """
    Crea o actualiza el vectorstore para una categoría específica.
    """
    category_dir = os.path.join(persist_directory, category)
    os.makedirs(category_dir, exist_ok=True)
    collection_name = COLLECTION_NAMES.get(category)
    checkpoint_file = os.path.join(category_dir, "checkpoint.json")
    
    try:
        vectorstore = Chroma(
            persist_directory=category_dir,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        
        total_chunks = len(chunks)
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            vectorstore.add_documents(batch)
            
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'processed_chunks': i + len(batch),
                    'total_chunks': total_chunks,
                    'last_update': datetime.now().isoformat()
                }, f)
            
            print(f"  💾 Guardado lote {i//batch_size + 1}: {len(batch)} chunks")
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error al procesar {category}: {e}")
        return None

def verify_collection_integrity(category_dir: str, collection_name: str, embeddings: VertexAIEmbeddings) -> Optional[Chroma]:
    """
    Verifica exhaustivamente la integridad de una colección Chroma existente.
    """
    try:
        required_base_files = ['chroma.sqlite3']
        missing_base = [f for f in required_base_files 
                       if not os.path.exists(os.path.join(category_dir, f))]
        if missing_base:
            print(f"  ⚠️ Faltan archivos base: {', '.join(missing_base)}")
            return None

        db_size = os.path.getsize(os.path.join(category_dir, 'chroma.sqlite3'))
        print(f"  📊 Tamaño de base de datos: {db_size/1024/1024:.2f} MB")
        
        if db_size < 1024:  # Menos de 1KB
            print(f"  ⚠️ Base de datos demasiado pequeña")
            return None

        try:
            vectorstore = Chroma(
                persist_directory=category_dir,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            
            collection_data = vectorstore.get()
            num_documents = len(collection_data['ids'])
            
            if num_documents == 0:
                print(f"  ⚠️ Colección vacía")
                return None
            
            try:
                results = vectorstore.similarity_search("test", k=1)
                print(f"  ✅ Colección funcional:")
                print(f"     - Documentos: {num_documents}")
                print(f"     - Búsqueda de prueba exitosa")
                return vectorstore
                
            except Exception as e:
                print(f"  ⚠️ Error en búsqueda de prueba: {str(e)}")
                return None

        except Exception as e:
            print(f"  ⚠️ Error accediendo a los datos: {str(e)}")
            return None

    except Exception as e:
        print(f"  ⚠️ Error en verificación: {str(e)}")
        return None

def initialize_vectorstore(pdf_directory: str, persist_directory: str) -> Dict[str, Chroma]:
    """
    Inicializa las colecciones de vectorstore, priorizando colecciones existentes y manejando la actualización incremental.
    """
    print("📚 Inicializando vectorstores...")
    
    embeddings = EmbeddingsManager.get_embeddings()
    vectorstores: Dict[str, Chroma] = {}
    
    if not os.path.exists(persist_directory):
        print(f"⚠️ Directorio de persistencia no existe: {persist_directory}")
        os.makedirs(persist_directory, exist_ok=True)
    
    for category in COLLECTION_NAMES:
        category_dir = os.path.join(persist_directory, category)
        collection_name = COLLECTION_NAMES[category]
        index_file = os.path.join(category_dir, "index.txt")
        processed_files: Dict[str, str] = {}
        
        print(f"\n🔍 Verificando colección: {category}")
        
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                for line in f:
                    try:
                        filepath, filehash = line.strip().split(",")
                        processed_files[filepath] = filehash
                    except ValueError:
                        print(f"  ⚠️ Línea inválida en index.txt: {line.strip()}")
                        continue
        
        if os.path.exists(category_dir):
            print(f"  📂 Encontrado directorio: {category_dir}")
            if vectorstore := verify_collection_integrity(category_dir, collection_name, embeddings):
                vectorstores[category] = vectorstore
                print(f"✅ Colección cargada exitosamente: {category}")
            else:
                print(f"⚠️ Se encontró directorio pero la colección no es válida: {category}")
                user_input = input(f"   ❓ ¿Desea intentar recrear esta colección? (s/n): ").lower()
                if user_input != 's':
                    print(f"   ⏩ Saltando recreación de: {category}")
                    continue
                import shutil
                shutil.rmtree(category_dir)
                os.makedirs(category_dir, exist_ok=True)
        
        else:
            print(f"  📂 No se encontró directorio para: {category}")
            user_input = input(f"   ❓ ¿Desea crear nueva colección para '{category}'? (s/n): ").lower()
            if user_input != 's':
                print(f"   ⏩ Saltando creación de: {category}")
                continue
        
        pdf_files_in_category = [f for f in os.listdir(os.path.join(pdf_directory, category)) if f.endswith(".pdf")]
        new_or_modified_files = []

        for pdf_file in pdf_files_in_category:
            pdf_path = os.path.join(pdf_directory, category, pdf_file)
            file_hash = _calculate_pdf_hash(pdf_path)
            if pdf_path not in processed_files or processed_files[pdf_path] != file_hash:
                new_or_modified_files.append(pdf_path)

        if not new_or_modified_files:
            print(f"  🔄 No hay archivos nuevos o modificados en {category}.")
            if category in vectorstores:
                continue
            else:
                vectorstores[category] = Chroma(
                    persist_directory=category_dir,
                    embedding_function=embeddings,
                    collection_name=collection_name
                )
                continue

        print(f"\n🔄 Iniciando procesamiento de colección: {category}")
        try:
            documents = load_category_documents(pdf_directory, category)
            if not documents:
                print(f"⚠️ No hay documentos para procesar en: {category}")
                continue
            
            chunks = process_documents_in_batches(documents)
            print(f"✅ Generados {len(chunks)} chunks")

            if vectorstore := create_category_vectorstore(chunks, category, embeddings, persist_directory):
                vectorstores[category] = vectorstore
                print(f"✅ Colección actualizada exitosamente: {category}")

                with open(index_file, "w") as f:
                    for pdf_file in pdf_files_in_category:
                        pdf_path = os.path.join(pdf_directory, category, pdf_file)
                        file_hash = _calculate_pdf_hash(pdf_path)
                        f.write(f"{pdf_path},{file_hash}\n")

        except Exception as e:
            print(f"❌ Error procesando {category}: {e}")
            continue
    
    if vectorstores:
        print("\n📊 Resumen de colecciones disponibles:")
        for category, vs in vectorstores.items():
            collection_size = len(vs.get()['ids'])
            print(f"   - {category}: {collection_size} documentos")
    else:
        print("\n⚠️ No se pudo cargar ninguna colección")
    
    return vectorstores

def diagnose_collection(category_dir: str) -> None:
    """
    Realiza un diagnóstico detallado de una colección existente.
    """
    print(f"\n🔍 Diagnóstico de colección en: {category_dir}")
    
    try:
        print("\n1. Estructura de archivos:")
        if os.path.exists(os.path.join(category_dir, 'chroma.sqlite3')):
            size_mb = os.path.getsize(os.path.join(category_dir, 'chroma.sqlite3')) / 1024 / 1024
            print(f"  ✅ chroma.sqlite3 encontrado ({size_mb:.2f} MB)")
        else:
            print("  ❌ chroma.sqlite3 no encontrado")

        print("\n2. Directorios de índice:")
        index_dirs = [d for d in os.listdir(category_dir) 
                     if os.path.isdir(os.path.join(category_dir, d))]
        for idx_dir in index_dirs:
            if len(idx_dir) == 36:  # UUID length
                print(f"  ✅ Directorio de índice encontrado: {idx_dir}")
                idx_path = os.path.join(category_dir, idx_dir)
                print("    Archivos en el directorio:")
                for f in os.listdir(idx_path):
                    size_kb = os.path.getsize(os.path.join(idx_path, f)) / 1024
                    print(f"    - {f} ({size_kb:.2f} KB)")

        print("\n3. Archivo de checkpoint:")
        checkpoint_path = os.path.join(category_dir, "checkpoint.json")
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                print(f"  ✅ Checkpoint encontrado:")
                print(f"    - Chunks procesados: {checkpoint_data.get('processed_chunks')}")
                print(f"    - Total chunks: {checkpoint_data.get('total_chunks')}")
                print(f"    - Última actualización: {checkpoint_data.get('last_update')}")
        else:
            print("  ⚠️ No se encontró archivo de checkpoint")

    except Exception as e:
        print(f"\n❌ Error durante el diagnóstico: {str(e)}")

def diagnose_all_collections(base_dir: str) -> None:
    """
    Diagnostica todas las colecciones en el directorio base.
    """
    for category in COLLECTION_NAMES:
        category_dir = os.path.join(base_dir, category)
        if os.path.exists(category_dir):
            diagnose_collection(category_dir)

def main():
    """Función principal para cargar PDFs y crear vectorstore."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pdf_directory = os.path.join(base_dir, "data", "raw", "pdf_docs")
    persist_directory = os.path.join(base_dir, "data", "processed", "vectorstores")
    
    if os.name == 'nt' and not os.path.exists(pdf_directory):
        pdf_directory = r"C:\Users\mfuen\OneDrive\Desktop\rag_docente_sin_UI\data\raw\pdf_docs"
    
    print(f"🔍 Directorio de PDFs: {pdf_directory}")
    print(f"💾 Directorio para persistencia: {persist_directory}")

    if not os.path.exists(pdf_directory):
        print(f"❌ El directorio de PDFs no existe: {pdf_directory}")
        return
    
    os.makedirs(persist_directory, exist_ok=True)
    
    vectorstores = initialize_vectorstore(pdf_directory, persist_directory)

    for category, vectorstore in vectorstores.items():
        collection_size = len(vectorstore.get()['ids'])
        print(f"✅ Vectorstore creado para {category} con {collection_size} documentos")

if __name__ == "__main__":
    main()
