from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils.rate_limiter import rate_limited_llm_call

def retrieve_with_filter(vectorstore, query, category=None, k=5):
    """
    Realiza una búsqueda en la vectorstore con filtro opcional por categoría.
    
    Args:
        vectorstore: La base de datos vectorial de Chroma
        query: Consulta para buscar
        category: Categoría para filtrar (opcional)
        k: Número de documentos a recuperar
        
    Returns:
        Lista de documentos relevantes
    """
    # Crear filtro si se especifica categoría
    filter_dict = None
    if category:
        if isinstance(category, list):
            filter_dict = {"category": {"$in": category}}
        else:
            filter_dict = {"category": {"$eq": category}}
    
    # Usar MMR para diversidad en resultados
    return vectorstore.max_marginal_relevance_search(
        query=query,
        k=k,
        fetch_k=k*2,  # Recuperar más para seleccionar los más diversos
        lambda_mult=0.7,  # Balance entre relevancia (1.0) y diversidad (0.0)
        filter=filter_dict
    )

def retrieve_documents(vectorstore, query, categories=None, k=5):
    """
    Recupera documentos relevantes para una consulta de manera estratégica.
    Prioriza documentos de diferentes categorías para obtener una visión completa.
    
    Args:
        vectorstore: La base de datos vectorial
        query: Consulta del usuario
        categories: Lista de categorías prioritarias (None = todas)
        k: Número total de documentos a recuperar
        
    Returns:
        Lista combinada de documentos relevantes
    """
    docs = []
    
    # Si no hay categorías específicas, hacer búsqueda general
    if not categories:
        docs = retrieve_with_filter(vectorstore, query, category=None, k=k)
        return docs
    
    # Distribuir la cantidad de documentos por categoría
    docs_per_category = max(1, k // len(categories))
    remaining = k
    
    # Recuperar documentos de cada categoría prioritaria
    for category in categories:
        category_docs = retrieve_with_filter(
            vectorstore, 
            query, 
            category=category, 
            k=min(docs_per_category, remaining)
        )
        docs.extend(category_docs)
        remaining -= len(category_docs)
        
        if remaining <= 0:
            break
    
    # Si quedan slots disponibles, hacer una búsqueda general para complementar
    if remaining > 0:
        general_docs = retrieve_with_filter(vectorstore, query, category=None, k=remaining)
        docs.extend(general_docs)
    
    return docs

def get_context_from_documents(docs, max_length=6000):
    """
    Extrae el contenido de los documentos y lo convierte en un contexto utilizable.
    
    Args:
        docs: Lista de documentos recuperados
        max_length: Longitud máxima del contexto
        
    Returns:
        Texto de contexto y lista de fuentes
    """
    if not docs:
        return "", []
    
    # Extraer contenido y metadata
    context_parts = []
    sources = []
    
    current_length = 0
    
    for doc in docs:
        # Añadir contenido si no excede el límite
        if current_length + len(doc.page_content) <= max_length:
            context_parts.append(doc.page_content)
            current_length += len(doc.page_content)
            
            # Registrar fuente si no está ya incluida
            if "source" in doc.metadata:
                source = doc.metadata["source"]
                if source not in sources:
                    sources.append(source)
                    
            # Añadir categoría a la fuente
            if "category" in doc.metadata:
                source_with_category = f"{doc.metadata.get('category')}: {doc.metadata.get('source', 'desconocido')}"
                if source_with_category not in sources:
                    sources.append(source_with_category)
        else:
            # Si excede el límite, parar
            break
    
    # Unir las partes del contexto
    context = "\n\n".join(context_parts)
    
    return context, sources 