import os
import sys
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
#from langchain_google_vertexai import VertexAIEmbeddings  # Eliminado
import json
from datetime import datetime
from langchain.schema import Document

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

class EmbeddingsManager:
    """Singleton para gestionar una única instancia de embeddings (implementación eliminada)."""
    @classmethod
    def get_embeddings(cls):
        raise NotImplementedError("La implementación de Google Cloud en embeddings ha sido eliminada.")

def split_large_document(doc, max_tokens: int = 15000):
    """Divide documentos grandes en fragmentos más pequeños."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents([doc])

def load_and_split_documents(pdf_directory: str, collection_name: str, max_tokens: int = 15000) -> List[Document]:
    """Carga y divide documentos desde un directorio, manejando diferentes tipos de archivos."""
    print(f"🔍 Verificando colección: {collection_name}")

    # Construcción CORRECTA de la ruta:
    collection_folder_name = collection_name.replace("pdf-rag-", "")  # Obtiene "propuesta" de "pdf-rag-propuesta"
    path_to_pdfs = os.path.join(pdf_directory, collection_folder_name) # Une data/raw con propuesta
    print(f"  📂 Buscando PDFs en: {path_to_pdfs}")  #  Línea de depuración

    loader = DirectoryLoader(
        path=path_to_pdfs,  # Usar la ruta construida
        glob="**/*.pdf",  # Busca PDFs recursivamente dentro de la carpeta
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )

    docs = loader.load()
    if not docs:
        print(f"  📂 No se encontraron archivos PDF en: {path_to_pdfs}")
        return []

    print(f"  📄 Documentos cargados: {len(docs)}")
    all_splits = []
    for doc in tqdm(docs, desc="Dividiendo documentos"):
        splits = split_large_document(doc, max_tokens=max_tokens)
        all_splits.extend(splits)
    print(f"  ✅ Documentos divididos en {len(all_splits)} fragmentos")
    return all_splits


def initialize_vectorstore(pdf_directory: str, persist_directory: str) -> Dict[str, Any]:
    """
    Inicializa el vectorstore, creando o cargando colecciones según sea necesario.
    """
    print("📚 Inicializando vectorstores...")
    # embeddings = EmbeddingsManager.get_embeddings() # Aunque de error, es necesario para la creacion #COMENTADO
    vectorstores = {}

    for category, collection_name in COLLECTION_NAMES.items():
        collection_path = os.path.join(persist_directory, collection_name)
        if os.path.exists(collection_path) and os.listdir(collection_path):
            print(f"  📂 Encontrado directorio: {collection_path}")
            try:
                # Cargar la colección existente, *pasando la función de embedding*
                vectorstore = Chroma(
                    collection_name=collection_name,
                    # embedding_function=embeddings,  #  embedding_function aquí #COMENTADO
                    persist_directory=persist_directory
                )
                vectorstores[category] = vectorstore
                print(f"  ✅ Colección '{category}' cargada.")

            except Exception as e:
                print(f"  ⚠️ Error al cargar la colección '{category}': {e}")
                print(f"   ❓ ¿Desea intentar recrear esta colección? (s/n): ", end="")
                if input().lower() == 's':
                    try:
                        print(f"  🗑️ Eliminando colección existente...")
                        import shutil
                        shutil.rmtree(collection_path)  # Eliminar directorio
                        print(f"  ✅ Colección eliminada.")
                    except Exception as e:
                        print(f"  ❌ Error al eliminar colección: {e}")
                        continue  # Saltar a la siguiente colección
                else:
                    print(f"   ⏩ Saltando recreación de: {category}")
                    continue # Saltar esta colección

        else: # No existe o está vacía
            print(f"  🔍 Verificando colección: {category}")
            docs = load_and_split_documents(pdf_directory, collection_name)
            if not docs:
                print(f"   ⏩ Saltando creación de: {category}")
                continue

            print(f"   ❓ ¿Desea crear nueva colección para '{category}'? (s/n): ", end="")
            if input().lower() != 's':
                print(f"   ⏩ Saltando creación de: {category}")
                continue

            try:
                vectorstore = Chroma.from_documents(
                    docs,
                    # embeddings, #COMENTADO
                    collection_name=collection_name,
                    persist_directory=persist_directory
                )
                vectorstore.persist()
                vectorstores[category] = vectorstore
                print(f"  ✅ Colección '{category}' creada y persistida.")
            except Exception as e:
                print(f"  ❌ Error al crear la colección '{category}': {e}")
                continue  # Saltar a la siguiente colección

    if not vectorstores:
        print("\n⚠️ No se pudo cargar ninguna colección")
    return vectorstores