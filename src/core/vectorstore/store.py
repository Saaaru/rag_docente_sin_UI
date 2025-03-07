from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_vectorstore(documents, embeddings, collection_name="pdf-rag-chroma"):
    """
    Crea o carga un Chroma vector store.
    Si la base de datos existe, simplemente la carga sin procesar nuevos documentos.
    Solo crea una nueva si no existe.
    """
    persist_directory = f"./{collection_name}"

    # Si existe la base de datos, cargarla y retornar
    if os.path.exists(persist_directory):
        print(
            f"\n📚 Base de datos Chroma existente encontrada en {persist_directory}")
        try:
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                collection_name=collection_name
            )
            collection_size = len(vectorstore.get()['ids'])
            print(
                f"✅ Vectorstore cargado exitosamente con {collection_size} documentos")
            return vectorstore
        except Exception as e:
            print(f"❌ Error crítico al cargar la base de datos existente: {e}")
            raise e

    # Solo si NO existe la base de datos, crear una nueva
    print("\n⚠️ No se encontró una base de datos existente. Creando nueva...")

    if not documents:
        raise ValueError(
            "❌ No se proporcionaron documentos para crear el vectorstore")

    print("📄 Procesando documentos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Documentos divididos en {len(chunks)} chunks")

    print("🔄 Creando nuevo vectorstore...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    collection_size = len(vectorstore.get()['ids'])
    print(f"✅ Nuevo vectorstore creado con {collection_size} documentos")

    return vectorstore