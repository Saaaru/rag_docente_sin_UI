from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.utils.rate_limiter import rate_limited_llm_call
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma

def retrieve_documents(
    vectorstores: Dict[str, Chroma],
    query: str,
    categories: Optional[List[str]] = None,
    k: int = 5
) -> List[Document]:
    """
    Recupera documentos relevantes de las colecciones especificadas.
    """
    if not vectorstores:
        return []

    # Usar todas las categorías si no se especifican
    search_categories = categories or list(vectorstores.keys())
    valid_categories = [cat for cat in search_categories if cat in vectorstores]

    results = []
    docs_per_category = max(1, k // len(valid_categories)) if valid_categories else 0

    # Buscar en cada categoría
    for category in valid_categories:
        try:
            category_results = vectorstores[category].max_marginal_relevance_search(
                query=query,
                k=docs_per_category,
                fetch_k=docs_per_category*2,
                lambda_mult=0.7
            )
            results.extend(category_results)
        except Exception as e:
            print(f"Error al buscar en {category}: {e}")
            continue

    return results[:k]

def get_context_from_documents(docs: List[Document], max_length: int = 6000) -> tuple[str, List[str]]:
    """
    Extrae el contexto y fuentes de los documentos recuperados.
    """
    if not docs:
        return "", []

    context_parts = []
    sources = []
    current_length = 0

    for doc in docs:
        if current_length + len(doc.page_content) <= max_length:
            context_parts.append(doc.page_content)
            current_length += len(doc.page_content)
            
            source = doc.metadata.get('source', '')
            if source and source not in sources:
                sources.append(source)

    return "\n\n".join(context_parts), sources

# Funciones nuevas para búsquedas por categoría

def retrieve_by_category(vectorstores: Dict[str, Chroma],
                         category: str,
                         query: str,
                         k: int = 5,
                         fetch_k_multiplier: int = 2,
                         lambda_mult: float = 0.7) -> List[Document]:
    """
    Función genérica para recuperar documentos de una categoría específica del vectorstore.
    
    Args:
        vectorstores: Diccionario de vectorstores.
        category: Nombre de la categoría a buscar.
        query: Consulta de búsqueda.
        k: Número de documentos a recuperar.
        fetch_k_multiplier: Factor para calcular fetch_k (default: 2).
        lambda_mult: Multiplicador lambda para el algoritmo de búsqueda (default: 0.7).
    
    Returns:
        Lista de documentos encontrados o lista vacía en caso de error.
    """
    if category not in vectorstores:
        print(f"❌ Categoría '{category}' no encontrada en los vectorstores disponibles")
        return []
    try:
        documents = vectorstores[category].max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=k * fetch_k_multiplier,
            lambda_mult=lambda_mult
        )
        print(f"✅ Se han recuperado {len(documents)} documentos de la categoría '{category}'")
        return documents
    except Exception as e:
        print(f"❌ Error al recuperar documentos de la categoría {category}: {e}")
        return []


def retrieve_bases_curriculares(vectorstores: Dict[str, Chroma], query: str, k: int = 5) -> List[Document]:
    """
    Recupera documentos de la categoría 'bases curriculares'.
    """
    return retrieve_by_category(vectorstores, "bases curriculares", query, k)


def retrieve_actividades_sugeridas(vectorstores: Dict[str, Chroma], query: str, k: int = 5) -> List[Document]:
    """
    Recupera documentos de la categoría 'actividades sugeridas'.
    """
    return retrieve_by_category(vectorstores, "actividades sugeridas", query, k)


def retrieve_orientaciones(vectorstores: Dict[str, Chroma], query: str, k: int = 5) -> List[Document]:
    """
    Recupera documentos de la categoría 'orientaciones'.
    """
    return retrieve_by_category(vectorstores, "orientaciones", query, k)


def retrieve_propuesta(vectorstores: Dict[str, Chroma], query: str, k: int = 5) -> List[Document]:
    """
    Recupera documentos de la categoría 'propuesta'.
    """
    return retrieve_by_category(vectorstores, "propuesta", query, k) 