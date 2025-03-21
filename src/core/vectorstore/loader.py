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

logger = logging.getLogger(__name__)

# Constantes optimizadas según los límites de Vertex AI
MAX_EMBEDDING_BATCH = 250000  
MAX_CHUNK_SIZE = 2000         
CHUNK_OVERLAP = 500           
BATCH_SIZE = 50               
MAX_RETRIES = 2               

# Información del modelo basada en model_config.py
print(f"📋 Configuración: Usando modelo de embeddings {EMBEDDING_MODEL_NAME}")
print(f"📋 Configuración: Usando modelo LLM {LLM_MODEL_NAME}")

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

print(f"⚙️ Usando configuración optimizada:")
print(f"   - Tamaño de fragmento: {MAX_CHUNK_SIZE} caracteres (~{MAX_CHUNK_SIZE//4} tokens)")
print(f"   - Solapamiento: {CHUNK_OVERLAP} caracteres")
print(f"   - Tamaño de lote: {BATCH_SIZE} documentos")

def split_large_document(doc: Document, max_tokens: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """Divide un documento en fragmentos más grandes optimizados."""
    if len(doc.page_content) <= max_tokens:
        doc.metadata.update({
            'is_split': False,
            'original_source': doc.metadata.get('source', '')
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
                'original_source': doc.metadata.get('source', ''),
                'parent_doc': os.path.basename(doc.metadata.get('source', 'unknown'))
            })
        return splits
    except Exception as e:
        print(f"⚠️ Error al dividir documento {doc.metadata.get('source', 'unknown')}: {e}")
        try:
            simpler_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_tokens // 2,
                chunk_overlap=overlap // 2,
                length_function=len,
            )
            return simpler_splitter.split_documents([doc])
        except:
            print(f"❌ No se pudo procesar el documento {doc.metadata.get('source', 'unknown')}")
            return []

def load_pdf_documents_from_folder(folder_path: str) -> List[Document]:
    """
    Busca recursivamente archivos PDF en la carpeta indicada y devuelve una lista de Document.
    """
    folder_path_str = str(folder_path)  # Convertimos a cadena
    pdf_files = glob(os.path.join(folder_path_str, '**', '*.pdf'), recursive=True)
    docs = []
    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(pdf)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error cargando {pdf}: {e}")
    return docs

def split_documents(docs: List[Document], chunk_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """
    Divide los documentos utilizando RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    splitted_docs = []
    for doc in docs:
        if len(doc.page_content) > chunk_size:
            splitted_docs.extend(splitter.split_documents([doc]))
        else:
            splitted_docs.append(doc)
    return splitted_docs

def initialize_vectorstores_simplified() -> Dict[str, Chroma]:
    """
    Recorre cada subcarpeta en RAW_DIR, carga los archivos PDF, los divide (si es necesario)
    y crea una colección en Chroma para cada subcarpeta, usando el nombre de la carpeta.
    """
    vectorstores = {}
    embeddings = get_embeddings()
    if not embeddings:
        print("Error: No se obtuvieron embeddings")
        return vectorstores

    # Convertimos RAW_DIR a cadena para evitar problemas de tipo
    raw_dir_str = str(RAW_DIR)
    for entry in os.listdir(raw_dir_str):
        subfolder_path = os.path.join(raw_dir_str, entry)
        if os.path.isdir(subfolder_path):
            print(f"\nProcesando carpeta: {entry}")
            docs = load_pdf_documents_from_folder(subfolder_path)
            print(f"Se encontraron {len(docs)} archivos PDF en {entry}")
            if not docs:
                continue
            docs = split_documents(docs)
            print(f"Después de dividir, {len(docs)} fragmentos en {entry}")
            try:
                vectorstore = Chroma(
                    collection_name=entry,
                    embedding_function=embeddings,
                    persist_directory=str(PERSIST_DIRECTORY)  # Convertir a cadena
                )
                vectorstore.add_documents(docs)
                vectorstores[entry] = vectorstore
                print(f"Colección '{entry}' creada con éxito.")
            except Exception as e:
                print(f"Error creando colección para {entry}: {e}")
    return vectorstores

def sanitize_collection_name(name: str) -> str:
    """Sanitiza el nombre de la colección."""
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    name = re.sub(r'[^a-zA-Z0-9-_]', '_', name)
    if name and name[0].isdigit():
        name = f"col_{name}"
    return name[:63]

def initialize_vectorstore(persist_directory: str = str(PERSIST_DIRECTORY)) -> Dict[str, Chroma]:
    """
    Inicializa vectorstores para cada subdirectorio en RAW_DIR.
    Esta versión simplificada carga los PDFs de cada carpeta, los divide y crea una colección en Chroma para cada una.
    """
    logger.info("Inicializando vectorstores (versión simplificada)...")
    return initialize_vectorstores_simplified()