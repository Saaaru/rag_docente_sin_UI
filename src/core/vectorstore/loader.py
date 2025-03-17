import os
from typing import List, Dict, Any
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from config.paths import RAW_DIR, PERSIST_DIRECTORY
from core.embeddings import get_embeddings  # Ahora importamos desde aquí
from pathlib import Path

# Constantes ajustadas para mantener más contexto
MAX_CHUNK_SIZE = 1000  # Tamaño más grande para mantener contexto
CHUNK_OVERLAP = 200    # Mayor solapamiento para mejor continuidad
BATCH_SIZE = 5        # Lotes más pequeños para mejor manejo de memoria
MAX_RETRIES = 3       # Número de intentos para chunks problemáticos

def split_large_document(doc: Document, max_tokens: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """Divide un documento manteniendo la coherencia del contenido."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap,
        length_function=len,
        # Separadores ordenados para mantener la coherencia del contenido
        separators=[
            "\n\n",     # Primero intenta dividir por párrafos
            "\n",       # Luego por líneas
            ". ",       # Luego por oraciones
            ", ",       # Luego por cláusulas
            " ",        # Finalmente por palabras si es necesario
        ]
    )
    
    try:
        splits = text_splitter.split_documents([doc])
        # Preservar metadata importante
        for split in splits:
            split.metadata.update({
                'original_source': doc.metadata.get('source', ''),
                'chunk_size': max_tokens,
                'overlap': overlap,
                'parent_doc': doc.metadata.get('source', '').split('/')[-1]
            })
        return splits
    except Exception as e:
        print(f"⚠️ Error al dividir documento {doc.metadata.get('source', 'unknown')}: {e}")
        # Intento de recuperación con chunks más pequeños solo si es necesario
        try:
            return text_splitter.split_documents([doc], chunk_size=max_tokens//2)
        except:
            print(f"❌ No se pudo procesar el documento incluso con chunks más pequeños")
            return []

def process_documents_in_batches(docs: List[Document], batch_size: int = BATCH_SIZE) -> List[Document]:
    """Procesa documentos en lotes con manejo inteligente de errores."""
    all_splits = []
    failed_docs = []
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        batch_splits = []
        
        for doc in tqdm(batch, desc=f"Procesando lote {i//batch_size + 1}/{len(docs)//batch_size + 1}"):
            retry_count = 0
            success = False
            
            while retry_count < MAX_RETRIES and not success:
                try:
                    # Intentar con diferentes tamaños de chunk si es necesario
                    current_size = MAX_CHUNK_SIZE - (retry_count * 200)
                    current_overlap = CHUNK_OVERLAP - (retry_count * 50)
                    
                    splits = split_large_document(doc, 
                                               max_tokens=current_size,
                                               overlap=current_overlap)
                    
                    if splits:
                        batch_splits.extend(splits)
                        success = True
                        print(f"  ✓ Documento procesado: {doc.metadata.get('source', 'unknown')}")
                    else:
                        raise ValueError("No se generaron splits")
                        
                except Exception as e:
                    retry_count += 1
                    if retry_count == MAX_RETRIES:
                        print(f"❌ No se pudo procesar el documento después de {MAX_RETRIES} intentos")
                        failed_docs.append(doc)
                    else:
                        print(f"⚠️ Reintentando con chunks más pequeños... ({retry_count}/{MAX_RETRIES})")
        
        all_splits.extend(batch_splits)
        print(f"  ✓ Lote {i//batch_size + 1} completado: {len(batch_splits)} fragmentos")
    
    if failed_docs:
        print(f"\n⚠️ {len(failed_docs)} documentos no pudieron ser procesados:")
        for doc in failed_docs:
            print(f"  - {doc.metadata.get('source', 'unknown')}")
    
    return all_splits

def create_collection_in_batches(
    docs: List[Document],
    embeddings,
    collection_name: str,
    persist_directory: str,
    batch_size: int = BATCH_SIZE
) -> Chroma:
    """Crea una colección de Chroma con manejo inteligente de errores."""
    try:
        # Inicializar la colección con un lote pequeño
        first_batch = docs[:batch_size]
        db = Chroma.from_documents(
            documents=first_batch,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        # Procesar el resto en lotes con reintentos
        remaining_docs = docs[batch_size:]
        failed_batches = []
        
        for i in range(0, len(remaining_docs), batch_size):
            batch = remaining_docs[i:i + batch_size]
            retry_count = 0
            success = False
            
            while retry_count < MAX_RETRIES and not success:
                try:
                    print(f"  ↳ Añadiendo lote {i//batch_size + 2} a '{collection_name}'")
                    db.add_documents(documents=batch)
                    success = True
                except Exception as e:
                    retry_count += 1
                    if retry_count == MAX_RETRIES:
                        print(f"❌ Fallo al añadir lote después de {MAX_RETRIES} intentos")
                        failed_batches.append(batch)
                    else:
                        print(f"⚠️ Reintentando añadir lote... ({retry_count}/{MAX_RETRIES})")
                        
        if failed_batches:
            print(f"\n⚠️ {len(failed_batches)} lotes no pudieron ser añadidos")
            
        return db
    except Exception as e:
        print(f"  ❌ Error al crear la colección: {e}")
        return None

def load_and_split_documents(directory: str, max_tokens: int = 1500, overlap: int = 200) -> List[Document]:
    """Carga y divide documentos PDF desde un directorio y sus subdirectorios.

    Args:
        directory: Directorio raíz a explorar (debe ser un string).
        max_tokens: Tamaño máximo de cada fragmento.
        overlap: Solapamiento entre fragmentos.

    Returns:
        Lista de documentos (fragmentos).
    """
    loader = DirectoryLoader(
        path=directory,
        glob="**/*.pdf",  # Busca PDFs recursivamente
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    try:
        docs = loader.load()
    except Exception as e:
        print(f"❌ Error al cargar documentos desde {directory}: {e}")
        return []

    if not docs:
        print(f"  📂 No se encontraron archivos PDF en {directory}")
        return []

    print(f"  📄 Documentos cargados: {len(docs)}")
    all_splits = []
    for doc in tqdm(docs, desc="Dividiendo documentos"):
        splits = split_large_document(doc, max_tokens=max_tokens, overlap=overlap)
        all_splits.extend(splits)
    print(f"  ✅ Documentos divididos en {len(all_splits)} fragmentos")
    return all_splits


def initialize_vectorstore(persist_directory: str = str(PERSIST_DIRECTORY)) -> Dict[str, Chroma]:
    """Inicializa el vectorstore, creando o cargando colecciones.

    Busca en subcarpetas de RAW_DIR, y crea una colección por subcarpeta.
    Si una colección ya existe, la carga; si no, la crea.

    Args:
        persist_directory: Directorio donde se guardan/cargan las colecciones.

    Returns:
        Diccionario de colecciones Chroma (nombre_coleccion: Chroma).
    """
    print("📚 Inicializando vectorstores...")

    # Crear el directorio de persistencia si no existe
    persist_dir_path = Path(persist_directory)
    persist_dir_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directorio de persistencia verificado/creado: {persist_dir_path.parent}")

    embeddings = get_embeddings()
    if embeddings is None:
        print(f"❌ Error: No se pudieron obtener los embeddings.")
        return {}

    vectorstores = {}
    raw_dir_str = str(RAW_DIR)

    # Verificar que RAW_DIR exista y tenga subdirectorios
    if not os.path.exists(raw_dir_str):
        print(f"❌ Error: El directorio {raw_dir_str} no existe.")
        return {}

    subdirs = [d for d in os.listdir(raw_dir_str) 
              if os.path.isdir(os.path.join(raw_dir_str, d))]
    
    if not subdirs:
        print(f"❌ Error: No se encontraron subdirectorios en {raw_dir_str}")
        return {}

    print(f"📁 Procesando {len(subdirs)} subdirectorios...")

    # Iterar sobre las subcarpetas de RAW_DIR
    for subdir_name in os.listdir(raw_dir_str):
        subdir_path = os.path.join(raw_dir_str, subdir_name)
        if os.path.isdir(subdir_path):  # Solo procesar subdirectorios
            collection_name = subdir_name  # El nombre de la subcarpeta es el nombre de la colección

            # --- Corrección: Sanitizar el nombre de la colección ---
            collection_name = collection_name.replace(" ", "_")  # Reemplaza espacios con guiones bajos
            collection_name = ''.join(c for c in collection_name if c.isalnum() or c == '_' or c == '-') #Elimina caracteres no permitidos
            if not (3 <= len(collection_name) <= 63):
                print(f"⚠️ Advertencia: Nombre de colección inválido después de sanitizar: '{collection_name}'.  Saltando.")
                continue
            if collection_name[0].isdigit() or collection_name[-1].isdigit(): #Asegura que no empiece ni termine con numeros
                print(f"⚠️ Advertencia: Nombre de colección inválido, empieza o termina con un número: '{collection_name}'.  Saltando.")
                continue
            # --- Fin de la corrección ---

            collection_path = os.path.join(persist_directory, collection_name)

            if os.path.exists(collection_path) and os.listdir(collection_path):
                print(f"  📂 Cargando colección existente '{collection_name}'...")
                try:
                    vectorstore = Chroma(
                        collection_name=collection_name,
                        embedding_function=embeddings,
                        persist_directory=persist_directory
                    )
                    vectorstores[collection_name] = vectorstore
                    print(f"  ✅ Colección '{collection_name}' cargada.")
                except Exception as e:
                    print(f"  ❌ Error al cargar la colección '{collection_name}': {e}")
                    # Podrías agregar una opción para recrear la colección aquí si falla la carga.
                    continue  # Saltar a la siguiente

            else:
                print(f"  📄 Creando colección '{collection_name}'...")
                docs = load_and_split_documents(subdir_path)
                if not docs:
                    print(f"   ⏩ Saltando creación de: '{collection_name}' (no hay documentos).")
                    continue

                try:
                    db = Chroma.from_documents(
                        documents=docs,
                        embedding=embeddings,
                        collection_name=collection_name,
                        persist_directory=persist_directory
                    )
                    vectorstores[collection_name] = db
                    print(f"  ✅ Colección '{collection_name}' creada.")
                except Exception as e:
                    print(f"  ❌ Error al crear la colección '{collection_name}': {e}")
                    continue  # Saltar a la siguiente

    if not vectorstores:
        print("\n⚠️ No se pudo cargar ni crear ninguna colección.")
    return vectorstores