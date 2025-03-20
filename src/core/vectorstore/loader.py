import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from src.config.paths import RAW_DIR, PERSIST_DIRECTORY
from src.core.embeddings import get_embeddings
from pathlib import Path
from src.config.model_config import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME
import logging
import re
import unicodedata
import chromadb

logger = logging.getLogger(__name__)

# Constantes optimizadas según los límites de Vertex AI
MAX_EMBEDDING_BATCH = 250000  # Aproximadamente 62,500 palabras para procesar a la vez
MAX_CHUNK_SIZE = 8000        # Aproximadamente 2,000 palabras por fragmento
CHUNK_OVERLAP = 800          # Aproximadamente 10% de solapamiento
BATCH_SIZE = 50              # Procesar más documentos por lote
MAX_RETRIES = 2              # Reducir los reintentos para agilizar el proceso

# Información del modelo basada en model_config.py
print(f"📋 Configuración: Usando modelo de embeddings {EMBEDDING_MODEL_NAME}")
print(f"📋 Configuración: Usando modelo LLM {LLM_MODEL_NAME}")

def get_optimal_chunk_settings() -> Dict[str, int]:
    """Determina los ajustes óptimos de fragmentación según el modelo."""
    # Para text-multilingual-embedding-002, ajustar según documentación
    if EMBEDDING_MODEL_NAME == "text-multilingual-embedding-002":
        return {
            "max_chunk_size": 8000,      # ~2000 palabras
            "chunk_overlap": 800,        # 10% de solapamiento
            "batch_size": 50             # 50 documentos por lote
        }
    # Para modelos textembedding-gecko
    elif "gecko" in EMBEDDING_MODEL_NAME:
        return {
            "max_chunk_size": 3000,      # ~750 palabras
            "chunk_overlap": 300,        # 10% de solapamiento
            "batch_size": 25             # Más conservador
        }
    # Default para otros modelos
    else:
        return {
            "max_chunk_size": MAX_CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "batch_size": BATCH_SIZE
        }

# Obtener configuración óptima
optimal_settings = get_optimal_chunk_settings()
MAX_CHUNK_SIZE = optimal_settings["max_chunk_size"]
CHUNK_OVERLAP = optimal_settings["chunk_overlap"]
BATCH_SIZE = optimal_settings["batch_size"]

print(f"⚙️ Usando configuración optimizada:")
print(f"   - Tamaño de fragmento: {MAX_CHUNK_SIZE} caracteres (~{MAX_CHUNK_SIZE//4} tokens)")
print(f"   - Solapamiento: {CHUNK_OVERLAP} caracteres") 
print(f"   - Tamaño de lote: {BATCH_SIZE} documentos")

def split_large_document(doc: Document, max_tokens: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """Divide un documento en fragmentos más grandes optimizados."""
    
    # Usar el texto completo si es suficientemente pequeño
    if len(doc.page_content) <= max_tokens:
        # Solo actualizar metadata
        doc.metadata.update({
            'is_split': False,
            'original_source': doc.metadata.get('source', ''),
        })
        return [doc]
    
    # Para documentos grandes, dividir de manera inteligente
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        length_function=len,
        separators=[
            "\n\n\n",    # Separaciones muy claras
            "\n\n",      # Párrafos
            "\n",        # Líneas
            ". ",        # Oraciones
            "? ",        # Preguntas
            "! ",        # Exclamaciones
            ", ",        # Cláusulas
            " ",         # Palabras
            ""           # Caracteres
        ]
    )
    
    try:
        splits = text_splitter.split_documents([doc])
        # Actualizar metadata
        for i, split in enumerate(splits):
            split.metadata.update({
                'is_split': True,
                'split_index': i,
                'total_splits': len(splits),
                'original_source': doc.metadata.get('source', ''),
                'parent_doc': os.path.basename(doc.metadata.get('source', 'unknown'))
            })
        return splits
    except Exception as e:
        print(f"⚠️ Error al dividir documento {doc.metadata.get('source', 'unknown')}: {e}")
        # Si falla, intentamos una división más simple
        try:
            simpler_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_tokens // 2,  # Reducir tamaño a la mitad
                chunk_overlap=overlap // 2,
                length_function=len,
            )
            return simpler_splitter.split_documents([doc])
        except:
            print(f"❌ No se pudo procesar el documento {doc.metadata.get('source', 'unknown')}")
            return []

def load_and_split_all_documents(directory: str) -> Dict[str, List[Document]]:
    """
    Carga TODOS los documentos de forma recursiva desde directory, los divide,
    y los agrupa por subdirectorio.

    Args:
        directory: La ruta base (RAW_DIR).

    Returns:
        Un diccionario donde las claves son los nombres de los subdirectorios
        (sanitizados) y los valores son listas de Document (ya divididos).
    """
    print(f"\n📂 Cargando y dividiendo documentos desde: {directory} (recursivamente)")

    loader = DirectoryLoader(
        path=directory,
        glob="**/*.pdf",  # Búsqueda recursiva
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )

    try:
        all_docs = loader.load()
        print(f"  ✅ Encontrados {len(all_docs)} documentos PDF (en total)")

        if not all_docs:
            print("  ⚠️ No se encontraron documentos PDF")
            return {}

        # Agrupar documentos por subdirectorio (antes de dividir)
        grouped_docs: Dict[str, List[Document]] = {}
        for doc in all_docs:
            # Obtener la ruta relativa al directorio base (RAW_DIR)
            relative_path = Path(doc.metadata['source']).relative_to(directory)
            # El primer elemento de la ruta relativa es el subdirectorio
            subdir_name = str(relative_path.parts[0])
            sanitized_subdir_name = sanitize_collection_name(subdir_name)

            if sanitized_subdir_name not in grouped_docs:
                grouped_docs[sanitized_subdir_name] = []
            grouped_docs[sanitized_subdir_name].append(doc)

        # Dividir documentos, manteniendo la agrupación por subdirectorio
        all_splits: Dict[str, List[Document]] = {}
        for subdir, docs in grouped_docs.items():
            print(f"\n  📁 Procesando subdirectorio: {subdir} ({len(docs)} documentos)")
            subdir_splits = []
            for doc in tqdm(docs, desc=f"Dividiendo documentos en {subdir}"):
                splits = split_large_document(doc)  # Usar la función de división
                subdir_splits.extend(splits)

            print(f"    📄 {subdir}: {len(docs)} documentos -> {len(subdir_splits)} fragmentos")
            all_splits[subdir] = subdir_splits

        return all_splits

    except Exception as e:
        print(f"❌ Error general al cargar/dividir documentos: {e}")
        return {}

def create_collection_in_batches(
    docs: List[Document],
    embeddings,
    collection_name: str,
    persist_directory: str,
) -> Optional[Chroma]:
    """Crea una colección de Chroma con manejo de lotes optimizado."""
    if not docs:
        print(f"⚠️ No hay documentos para crear la colección '{collection_name}'")
        return None
        
    doc_count = len(docs)
    print(f"\n🔨 Creando colección '{collection_name}' con {doc_count} documentos")
    
    try:
        # Crear colección vacía primero
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
            client=chroma_client
        )
        
        # Añadir documentos en lotes
        batch_size = min(BATCH_SIZE, max(1, doc_count // 10))  # Ajustar batch_size dinámicamente
        total_batches = (doc_count + batch_size - 1) // batch_size
        
        print(f"  📦 Procesando en {total_batches} lotes de ~{batch_size} documentos cada uno")
        
        for i in range(0, doc_count, batch_size):
            batch = docs[i:i + batch_size]
            batch_number = (i // batch_size) + 1
            
            try:
                print(f"  ↳ Añadiendo lote {batch_number}/{total_batches} ({len(batch)} documentos)")
                db.add_documents(documents=batch)
                print(f"    ✓ Lote {batch_number} completado")
            except Exception as e:
                print(f"    ❌ Error en lote {batch_number}: {str(e)}")
                # Intentar añadir los documentos uno por uno
                for j, doc in enumerate(batch):
                    try:
                        db.add_documents(documents=[doc])
                    except:
                        print(f"      ❌ No se pudo añadir el documento {j+1} del lote {batch_number}")
        
        # Verificar el resultado
        try:
            final_count = len(db.get()['ids'])
            print(f"\n✅ Colección '{collection_name}' creada con {final_count} documentos")
            return db
        except Exception as e:
            print(f"⚠️ No se pudo verificar el conteo final: {e}")
            return db
            
    except Exception as e:
        print(f"❌ Error al crear la colección: {e}")
        return None

def sanitize_collection_name(name: str) -> str:
    """Sanitiza el nombre de la colección."""
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    name = re.sub(r'[^a-zA-Z0-9-_]', '_', name)
    if name[0].isdigit():
        name = f"col_{name}"
    return name[:63]

def initialize_vectorstore(persist_directory: str = str(PERSIST_DIRECTORY)) -> Dict[str, Chroma]:
    """
    Inicializa vectorstores para cada subdirectorio en RAW_DIR, *solo si no existen*.
    """
    logger.info("Inicializando vectorstores...")

    embeddings = get_embeddings()
    if not embeddings:
        logger.error("No se pudieron obtener los embeddings")
        return {}

    # Cargar y dividir *todos* los documentos, agrupados por subdirectorio
    all_splits = load_and_split_all_documents(RAW_DIR)

    if not all_splits:
        logger.warning("No se encontraron documentos o hubo un error al cargarlos.")
        return {}

    vectorstores = {}

    # 1. Usar el cliente de Chroma para obtener las colecciones existentes
    try:
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        existing_collections = chroma_client.list_collections()
        existing_collection_names = [col.name for col in existing_collections]
        logger.info(f"Colecciones existentes: {existing_collection_names}")
    except Exception as e:
        logger.error(f"Error al listar colecciones existentes: {e}. Se asumirá que no hay ninguna.")
        existing_collection_names = []

    # Iterar sobre los subdirectorios y sus documentos (ya divididos)
    for subdir_name, split_docs in all_splits.items():
        # 2. Verificar si la colección ya existe (usando la lista obtenida)
        if subdir_name in existing_collection_names:
            logger.info(f"Colección '{subdir_name}' ya existe. Cargando...")
            try:
                vectorstore = Chroma(
                    collection_name=subdir_name,
                    embedding_function=embeddings,
                    persist_directory=persist_directory,
                    client=chroma_client,
                )
                vectorstores[subdir_name] = vectorstore
                logger.info(f"Colección '{subdir_name}' cargada.")
            except Exception as e:
                logger.error(f"Error al cargar la colección '{subdir_name}': {e}")
            continue  # Saltar al siguiente subdirectorio

        logger.info(f"Creando colección para: {subdir_name}")

        # 3. Crear y añadir a la colección (usando langchain_chroma)
        vectorstore = create_collection_in_batches(
            split_docs, embeddings, subdir_name, persist_directory
        )
        if vectorstore:
             vectorstores[subdir_name] = vectorstore

    return vectorstores