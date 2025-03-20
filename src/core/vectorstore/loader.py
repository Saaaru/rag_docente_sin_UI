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

def load_and_split_documents(directory: str) -> List[Document]:
    """Carga y divide documentos con logging mejorado."""
    print(f"\n📂 Cargando documentos desde: {directory}")
    
    # Sanitizar la ruta
    safe_directory = directory.replace(" ", "_")
    if safe_directory != directory:
        try:
            # Solo intentar renombrar si el directorio destino no existe
            if not os.path.exists(safe_directory):
                os.rename(directory, safe_directory)
                directory = safe_directory
                print(f"✓ Directorio renombrado: {directory}")
            else:
                print(f"⚠️ No se puede renombrar, ya existe: {safe_directory}")
        except Exception as e:
            print(f"⚠️ No se pudo renombrar el directorio: {e}")
    
    # Configurar el loader
    loader = DirectoryLoader(
        path=directory,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    
    try:
        print(f"  🔍 Buscando archivos PDF en {directory}...")
        docs = loader.load()
        doc_count = len(docs)
        
        if doc_count == 0:
            print("  ⚠️ No se encontraron documentos PDF")
            return []
        
        print(f"  ✅ Encontrados {doc_count} documentos PDF")
        
        # Ordenar documentos por tamaño (primero los pequeños)
        docs.sort(key=lambda doc: len(doc.page_content))
        
        # Procesar documentos en lotes paralelos
        all_splits = []
        counter = {"processed": 0, "failed": 0, "unchanged": 0, "split": 0}
        
        for doc in tqdm(docs, desc="Procesando documentos"):
            try:
                # Determinar si necesita dividirse o puede procesarse completo
                doc_len = len(doc.page_content)
                doc_tokens = doc_len // 4  # Estimación de tokens
                
                if doc_tokens <= MAX_CHUNK_SIZE // 4:  # Si cabe en un fragmento
                    # No dividir, usar documento completo
                    all_splits.append(doc)
                    counter["unchanged"] += 1
                    print(f"  ⏩ Documento pequeño, no requiere división: {os.path.basename(doc.metadata.get('source', 'unknown'))}")
                else:
                    # Dividir documento grande
                    splits = split_large_document(doc)
                    split_count = len(splits)
                    
                    if split_count > 0:
                        all_splits.extend(splits)
                        counter["split"] += 1
                        counter["processed"] += split_count
                        print(f"  ✂️ Documento dividido en {split_count} fragmentos: {os.path.basename(doc.metadata.get('source', 'unknown'))}")
                    else:
                        counter["failed"] += 1
                
            except Exception as e:
                print(f"  ❌ Error procesando: {os.path.basename(doc.metadata.get('source', 'unknown'))}: {e}")
                counter["failed"] += 1
        
        # Resumen
        total_splits = len(all_splits)
        print(f"\n📊 Resumen de procesamiento:")
        print(f"   - Documentos originales: {doc_count}")
        print(f"   - Documentos sin dividir: {counter['unchanged']}")
        print(f"   - Documentos divididos: {counter['split']}")
        print(f"   - Documentos con error: {counter['failed']}")
        print(f"   - Total fragmentos resultantes: {total_splits}")
        
        return all_splits
        
    except Exception as e:
        print(f"❌ Error general al cargar documentos: {str(e)}")
        return []

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

    vectorstores = {}
    raw_dir = Path(RAW_DIR)

    if not raw_dir.exists():
        logger.error(f"No existe el directorio {raw_dir}")
        return {}

    # 1. Usar el cliente de Chroma para obtener las colecciones existentes
    try:
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        existing_collections = chroma_client.list_collections()
        existing_collection_names = [col.name for col in existing_collections]
        logger.info(f"Colecciones existentes: {existing_collection_names}")
    except Exception as e:
        logger.error(f"Error al listar colecciones existentes: {e}. Se asumirá que no hay ninguna.")
        existing_collection_names = []


    for subdir in raw_dir.iterdir():
        if not subdir.is_dir():
            continue

        original_name = subdir.name
        collection_name = sanitize_collection_name(original_name)

        # 2. Verificar si la colección ya existe (usando la lista obtenida)
        if collection_name in existing_collection_names:
            logger.info(f"Colección '{collection_name}' ya existe. Cargando...")
            try:
                # Cargar la colección existente *usando langchain_chroma*
                vectorstore = Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    persist_directory=persist_directory,
                    client=chroma_client
                )
                vectorstores[collection_name] = vectorstore
                logger.info(f"Colección '{collection_name}' cargada.")
            except Exception as e:
                logger.error(f"Error al cargar la colección '{collection_name}': {e}")
            continue  # Saltar al siguiente subdirectorio

        if collection_name != original_name:
            logger.info(f"Nombre sanitizado: '{original_name}' -> '{collection_name}'")

        logger.info(f"Procesando colección: {collection_name}")

        try:
            loader = DirectoryLoader(
                str(subdir), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True
            )
            documents = loader.load()
            if not documents:
                logger.warning(f"No se encontraron documentos en {subdir}")
                continue

            # Usar la función de división
            split_docs = load_and_split_documents(str(subdir))

            logger.info(f"Documentos divididos en {len(split_docs)} fragmentos")

            for doc in split_docs:
                if 'source' in doc.metadata:
                    doc.metadata['source'] = sanitize_collection_name(doc.metadata['source'])
                doc.metadata['collection'] = collection_name

            # 3. Crear vectorstore *solo* si no existe (usando langchain_chroma)
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_directory,
                client=chroma_client
            )

            # 4. Procesamiento por lotes (solo si es una colección nueva)
            MAX_BATCH_SIZE = 5000
            num_batches = (len(split_docs) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
            logger.info(f"Añadiendo {len(split_docs)} documentos en {num_batches} lotes...")

            for i in range(0, len(split_docs), MAX_BATCH_SIZE):
                batch = split_docs[i:i + MAX_BATCH_SIZE]
                try:
                    vectorstore.add_documents(batch)  # Usar add_documents de langchain_chroma
                    logger.info(f"Lote {i // MAX_BATCH_SIZE + 1}/{num_batches} añadido.")
                except Exception as e:
                    logger.error(f"Error en lote {i // MAX_BATCH_SIZE + 1}: {e}")

            vectorstores[collection_name] = vectorstore

        except Exception as e:
            logger.error(f"Error procesando {collection_name}: {e}")
            continue

    return vectorstores