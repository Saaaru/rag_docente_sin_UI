from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils.rate_limiter import rate_limited_llm_call
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma

def retrieve_with_filter(vectorstores: Dict[str, Chroma], 
                        query: str, 
                        categories: Optional[List[str]] = None,
                        k: int = 5) -> List[Document]:
    """
    Realiza una búsqueda en las colecciones especificadas.
    
    Args:
        vectorstores: Diccionario de vectorstores por categoría
        query: Consulta para buscar
        categories: Lista de categorías donde buscar (None = todas)
        k: Número de documentos a recuperar por categoría
        
    Returns:
        Lista combinada de documentos relevantes
    """
    if not vectorstores:
        print("❌ No hay vectorstores disponibles")
        return []
    
    # Si no se especifican categorías, usar todas
    categories = categories or list(vectorstores.keys())
    
    # Validar categorías solicitadas
    valid_categories = [cat for cat in categories if cat in vectorstores]
    if not valid_categories:
        print("❌ No se encontraron categorías válidas para la búsqueda")
        return []
    
    if len(valid_categories) < len(categories):
        missing = set(categories) - set(valid_categories)
        print(f"⚠️ Categorías no encontradas: {', '.join(missing)}")
    
    results = []
    docs_per_category = max(1, k // len(valid_categories))
    
    # Realizar búsqueda en cada categoría
    for category in valid_categories:
        try:
            category_results = vectorstores[category].max_marginal_relevance_search(
                query=query,
                k=docs_per_category,
                fetch_k=docs_per_category*2,
                lambda_mult=0.7
            )
            results.extend(category_results)
            print(f"✅ {len(category_results)} documentos recuperados de {category}")
        except Exception as e:
            print(f"❌ Error al buscar en {category}: {e}")
    
    return results

def retrieve_documents(vectorstores: Dict[str, Chroma], 
                      query: str,
                      categories: Optional[List[str]] = None,
                      k: int = 5) -> List[Document]:
    """
    Recupera documentos relevantes de manera estratégica entre las colecciones.
    
    Args:
        vectorstores: Diccionario de vectorstores por categoría
        query: Consulta del usuario
        categories: Lista de categorías prioritarias (None = todas)
        k: Número total de documentos a recuperar
        
    Returns:
        Lista combinada de documentos relevantes
    """
    if not vectorstores:
        print("❌ No hay vectorstores disponibles")
        return []
    
    # Si no hay categorías específicas, hacer búsqueda en todas
    if not categories:
        return retrieve_with_filter(vectorstores, query, None, k)
    
    docs = []
    remaining = k
    
    # Primero buscar en las categorías prioritarias
    priority_results = retrieve_with_filter(
        vectorstores,
        query,
        categories,
        k=remaining
    )
    docs.extend(priority_results)
    remaining -= len(priority_results)
    
    # Si quedan slots y hay otras categorías disponibles, complementar
    if remaining > 0:
        other_categories = [cat for cat in vectorstores.keys() if cat not in categories]
        if other_categories:
            additional_results = retrieve_with_filter(
                vectorstores,
                query,
                other_categories,
                k=remaining
            )
            docs.extend(additional_results)
    
    return docs

def get_context_from_documents(docs: List[Document], max_length: int = 6000) -> tuple[str, List[str]]:
    """
    Extrae el contenido de los documentos y lo convierte en contexto utilizable.
    
    Args:
        docs: Lista de documentos recuperados
        max_length: Longitud máxima del contexto
        
    Returns:
        Tuple[str, List[str]]: (Texto de contexto, Lista de fuentes)
    """
    if not docs:
        return "", []
    
    context_parts = []
    sources = []
    current_length = 0
    
    for doc in docs:
        # Añadir contenido si no excede el límite
        if current_length + len(doc.page_content) <= max_length:
            context_parts.append(doc.page_content)
            current_length += len(doc.page_content)
            
            # Registrar fuente con categoría
            if "source" in doc.metadata and "category" in doc.metadata:
                source = f"{doc.metadata['category']}: {doc.metadata['source']}"
                if source not in sources:
                    sources.append(source)
            elif "source" in doc.metadata:
                if doc.metadata["source"] not in sources:
                    sources.append(doc.metadata["source"])
        else:
            break
    
    context = "\n\n".join(context_parts)
    return context, sources 

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