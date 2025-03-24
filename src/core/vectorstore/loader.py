import os
from glob import glob
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from src.config.paths import RAW_DIR, PERSIST_DIRECTORY
from src.core.embeddings import get_embeddings
from src.config.model_config import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME
import logging
import re
import unicodedata
import chromadb
import uuid

# Configurar logging
logger = logging.getLogger(__name__)

# Constantes optimizadas según los límites de Vertex AI
MAX_EMBEDDING_BATCH = 250000  
MAX_CHUNK_SIZE = 2000         
CHUNK_OVERLAP = 500           
BATCH_SIZE = 50               
MAX_RETRIES = 2               
MIN_TEXT_LENGTH = 50  # Umbral mínimo de caracteres para considerar una página útil

# Información del modelo basada en model_config.py
logger.info(f"📋 Configuración: Usando modelo de embeddings {EMBEDDING_MODEL_NAME}")
logger.info(f"📋 Configuración: Usando modelo LLM {LLM_MODEL_NAME}")

def get_optimal_chunk_settings() -> Dict[str, int]:
    """Determina los ajustes óptimos de fragmentación según el modelo."""
    if EMBEDDING_MODEL_NAME == "text-multilingual-embedding-002":
        return {
            "max_chunk_size": 2000,
            "chunk_overlap": 500,
            "batch_size": 50
        }
    elif "gecko" in EMBEDDING_MODEL_NAME:
        return {
            "max_chunk_size": 3000,
            "chunk_overlap": 300,
            "batch_size": 25
        }
    else:
        return {
            "max_chunk_size": MAX_CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "batch_size": BATCH_SIZE
        }

optimal_settings = get_optimal_chunk_settings()
MAX_CHUNK_SIZE = optimal_settings["max_chunk_size"]
CHUNK_OVERLAP = optimal_settings["chunk_overlap"]
BATCH_SIZE = optimal_settings["batch_size"]

logger.info("⚙️ Usando configuración optimizada:")
logger.info(f"   - Tamaño de fragmento: {MAX_CHUNK_SIZE} caracteres (~{MAX_CHUNK_SIZE//4} tokens)")
logger.info(f"   - Solapamiento: {CHUNK_OVERLAP} caracteres")
logger.info(f"   - Tamaño de lote: {BATCH_SIZE} documentos")

def split_large_document(doc: Document, max_tokens: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """Divide un documento en fragmentos más grandes optimizados."""
    if len(doc.page_content) <= max_tokens:
        doc.metadata.update({
            'is_split': False,
            'original_source': doc.metadata.get('source', doc.metadata.get('file_path', ''))
        })
        return [doc]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        length_function=len,
        separators=[
            "\n\n\n", 
            "\n\n",  
            "\n",    
            ". ",    
            "? ",    
            "! ",    
            ", ",    
            " ",     
            ""       
        ]
    )
    
    try:
        splits = text_splitter.split_documents([doc])
        for i, split in enumerate(splits):
            split.metadata.update({
                'is_split': True,
                'split_index': i,
                'total_splits': len(splits),
                'original_source': doc.metadata.get('source', doc.metadata.get('file_path', '')),
                'parent_doc': os.path.basename(doc.metadata.get('source', doc.metadata.get('file_path', 'unknown'))),
                'doc_id': str(uuid.uuid4())  # Añadimos un ID único para cada fragmento
            })
        return splits
    except Exception as e:
        logger.error(f"⚠️ Error al dividir documento {doc.metadata.get('source', 'unknown')}: {e}")
        try:
            simpler_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_tokens // 2,
                chunk_overlap=overlap // 2,
                length_function=len,
            )
            return simpler_splitter.split_documents([doc])
        except Exception as ex:
            logger.error(f"❌ No se pudo procesar el documento {doc.metadata.get('source', 'unknown')}: {ex}")
            return []

def load_pdf_documents_from_folder(folder_path: str) -> List[Document]:
    """
    Busca recursivamente archivos PDF en la carpeta indicada y devuelve una lista de Document.
    Solo se conservará el texto seleccionable (ignorando páginas con muy poco texto).
    """
    folder_path_str = str(folder_path)
    pdf_files = glob(os.path.join(folder_path_str, '**', '*.pdf'), recursive=True)
    
    logger.info(f"🔍 Encontrados {len(pdf_files)} archivos PDF en {folder_path_str}")
    
    docs = []
    for pdf in tqdm(pdf_files, desc="Cargando PDFs", unit="archivo"):
        try:
            loader = PyPDFLoader(pdf)
            loaded_docs = loader.load()
            
            # Añadir metadatos adicionales y garantizar que 'source' esté definido
            for doc in loaded_docs:
                doc.metadata.update({
                    'file_path': pdf,
                    'file_name': os.path.basename(pdf),
                    'file_type': 'pdf',
                    'category': os.path.basename(os.path.dirname(pdf)),
                    'source': pdf,
                    'doc_id': str(uuid.uuid4())
                })
            
            # Filtrar documentos con muy poco texto (menos de MIN_TEXT_LENGTH caracteres)
            filtered_docs = []
            for doc in loaded_docs:
                texto = doc.page_content.strip()
                if len(texto) < MIN_TEXT_LENGTH:
                    logger.debug(f"📄 Se descarta página con poco texto en {pdf}: {len(texto)} caracteres")
                    continue
                filtered_docs.append(doc)
            
            docs.extend(filtered_docs)
            logger.info(f"✅ Cargado: {pdf} - {len(filtered_docs)} páginas útiles")
        except Exception as e:
            logger.error(f"❌ Error cargando {pdf}: {e}")
    
    return docs

def split_documents(docs: List[Document], chunk_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """
    Divide los documentos utilizando RecursiveCharacterTextSplitter con manejo de error mejorado.
    Se descartan fragmentos con muy poco texto (menos de MIN_TEXT_LENGTH caracteres) para asegurar la relevancia.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    
    splitted_docs = []
    for doc in tqdm(docs, desc="Dividiendo documentos", unit="doc"):
        try:
            texto = doc.page_content.strip()
            if len(texto) < MIN_TEXT_LENGTH:
                logger.warning(f"⚠️ Documento con poco texto detectado: {doc.metadata.get('source', 'unknown')} (solo {len(texto)} caracteres)")
                continue
                
            if len(texto) > chunk_size:
                doc_splits = splitter.split_documents([doc])
                doc_splits = [split for split in doc_splits if len(split.page_content.strip()) >= MIN_TEXT_LENGTH]
                
                if doc_splits:
                    splitted_docs.extend(doc_splits)
                else:
                    logger.warning(f"⚠️ La división produjo fragmentos vacíos para: {doc.metadata.get('source', 'unknown')}")
            else:
                splitted_docs.append(doc)
        except Exception as e:
            logger.error(f"❌ Error al dividir documento {doc.metadata.get('source', 'unknown')}: {e}")
    
    logger.info(f"📊 Procesamiento completado: {len(splitted_docs)} fragmentos generados de {len(docs)} documentos")
    
    empty_docs = [doc for doc in splitted_docs if len(doc.page_content.strip()) < MIN_TEXT_LENGTH]
    if empty_docs:
        logger.warning(f"⚠️ Advertencia: {len(empty_docs)} fragmentos están vacíos o contienen muy poco texto y serán eliminados")
        splitted_docs = [doc for doc in splitted_docs if len(doc.page_content.strip()) >= MIN_TEXT_LENGTH]
    
    return splitted_docs

def sanitize_collection_name(name: str) -> str:
    """Sanitiza el nombre de la colección para ChromaDB."""
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    name = re.sub(r'[^a-zA-Z0-9-_]', '_', name)
    if name and name[0].isdigit():
        name = f"col_{name}"
    return name[:63]

def initialize_vectorstores_improved() -> Dict[str, Chroma]:
    """
    Versión mejorada que carga PDFs de cada carpeta, los divide y crea o carga una colección en Chroma
    con mejor manejo de errores y validación de documentos.
    """
    vectorstores = {}
    embeddings = get_embeddings()
    if not embeddings:
        logger.error("❌ Error: No se obtuvieron embeddings")
        return vectorstores

    persist_dir = str(PERSIST_DIRECTORY)
    os.makedirs(persist_dir, exist_ok=True)
    logger.info(f"📁 Usando directorio de persistencia: {persist_dir}")

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    existing_collections = chroma_client.list_collections()  # Ahora retorna una lista de nombres
    logger.info(f"💾 Colecciones existentes: {existing_collections}")

    raw_dir_str = str(RAW_DIR)
    for entry in os.listdir(raw_dir_str):
        subfolder_path = os.path.join(raw_dir_str, entry)
        if os.path.isdir(subfolder_path):
            try:
                logger.info(f"\n📁 Procesando carpeta: {entry}")
                collection_name = sanitize_collection_name(entry)
                logger.info(f"🏷️ Nombre de colección sanitizado: {collection_name}")
                
                # Si la colección ya existe, se carga para uso sin eliminarla
                if collection_name in existing_collections:
                    logger.info(f"💾 La colección '{collection_name}' ya existe. Se cargará para uso.")
                    vectorstore = Chroma(
                        collection_name=collection_name,
                        embedding_function=embeddings,
                        persist_directory=persist_dir
                    )
                    vectorstores[entry] = vectorstore
                    continue
                
                # Para una colección nueva, creamos la instancia de vectorstore
                vectorstore = Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    persist_directory=persist_dir
                )
                
                docs = load_pdf_documents_from_folder(subfolder_path)
                logger.info(f"📄 Se encontraron {len(docs)} documentos útiles en '{entry}'")
                
                if not docs:
                    logger.warning(f"⚠️ No se encontraron documentos útiles en {entry}, omitiendo...")
                    continue
                
                processed_docs = split_documents(docs)
                logger.info(f"✂️ Después de dividir: {len(processed_docs)} fragmentos")
                
                processed_docs = [doc for doc in processed_docs if len(doc.page_content.strip()) >= MIN_TEXT_LENGTH]
                if not processed_docs:
                    logger.error(f"❌ Error: No se generaron fragmentos válidos para {entry}")
                    continue
                
                if processed_docs:
                    first_doc = processed_docs[0]
                    logger.info("📝 Primer documento:")
                    logger.info(f"   - Longitud: {len(first_doc.page_content)} caracteres")
                    logger.info(f"   - Metadatos: {first_doc.metadata}")
                
                try:
                    total_docs = len(processed_docs)
                    logger.info(f"🔄 Agrupando documentos en lotes dinámicos para indexar {total_docs} fragmentos")
                    # Configuración para el agrupamiento dinámico:
                    MAX_ALLOWED_BATCH_TOKENS = 15000  # Límite ajustado a 15,000 tokens
                    average_chars_per_token = 4        # Aproximación: 4 caracteres por token
                    
                    dynamic_batch = []
                    dynamic_batch_tokens = 0
                    
                    for doc in processed_docs:
                        tokens_estimate = len(doc.page_content) / average_chars_per_token
                        if dynamic_batch_tokens + tokens_estimate > MAX_ALLOWED_BATCH_TOKENS:
                            logger.info(f"   - Añadiendo lote dinámico de {len(dynamic_batch)} documentos (≈{int(dynamic_batch_tokens)} tokens)")
                            vectorstore.add_documents(dynamic_batch)
                            dynamic_batch = [doc]
                            dynamic_batch_tokens = tokens_estimate
                        else:
                            dynamic_batch.append(doc)
                            dynamic_batch_tokens += tokens_estimate
                    
                    if dynamic_batch:
                        logger.info(f"   - Añadiendo último lote dinámico de {len(dynamic_batch)} documentos (≈{int(dynamic_batch_tokens)} tokens)")
                        vectorstore.add_documents(dynamic_batch)
                    
                    # Persistir la colección luego de agregar todos los documentos
                    try:
                        vectorstore.persist()
                        logger.info(f"💾 Persistencia de la colección '{collection_name}' completada")
                    except Exception as perr:
                        logger.warning(f"⚠️ No se pudo persistir la colección '{collection_name}': {perr}")
                    
                    vectorstores[entry] = vectorstore
                    logger.info(f"✅ Colección '{collection_name}' creada con éxito con {total_docs} documentos")
                
                except Exception as e:
                    logger.error(f"❌ Error creando colección para {entry}: {e}")
                    if "400" in str(e) and "text content is empty" in str(e).lower():
                        logger.warning("⚠️ Error de contenido vacío. Verificando documentos...")
                        for idx, doc in enumerate(processed_docs):
                            if len(doc.page_content.strip()) < MIN_TEXT_LENGTH:
                                logger.warning(f"   - Documento #{idx} contiene muy poco texto")
            
            except Exception as e:
                logger.error(f"❌ Error procesando carpeta {entry}: {e}")
    
    return vectorstores

def initialize_vectorstore() -> Dict[str, Chroma]:
    """
    Inicializa vectorstores para cada subdirectorio en RAW_DIR.
    """
    logger.info("Inicializando vectorstores (versión mejorada)...")
    logger.info("🚀 Inicializando vectorstores con ChromaDB en carpeta local")
    return initialize_vectorstores_improved()

class VectorstoreInitializationError(Exception):
    """Excepción personalizada para errores de inicialización del vectorstore."""
    pass
