from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.utils.rate_limiter import rate_limited_llm_call
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Configurar logging
logger = logging.getLogger(__name__)

def retrieve_documents(
    vectorstores: Dict[str, Chroma],
    query: str,
    categories: Optional[List[str]] = None,
    k: int = 5,
    fetch_k_multiplier: int = 2,
    lambda_mult: float = 0.7
) -> List[Document]:
    """
    Recupera documentos relevantes de las colecciones especificadas con manejo mejorado de errores.
    """
    if not vectorstores:
        logger.warning("No hay vectorstores disponibles para la búsqueda")
        return []

    search_categories = categories or list(vectorstores.keys())
    valid_categories = [cat for cat in search_categories if cat in vectorstores]
    
    if not valid_categories:
        logger.warning(f"Ninguna categoría válida encontrada. Categorías disponibles: {list(vectorstores.keys())}")
        return []

    results = []
    docs_per_category = max(1, k // len(valid_categories)) if valid_categories else 0
    fetch_k = docs_per_category * fetch_k_multiplier
    
    logger.info(f"Buscando en {len(valid_categories)} categorías: {valid_categories}")
    logger.info(f"Documentos por categoría: {docs_per_category}, fetch_k: {fetch_k}")

    for category in valid_categories:
        try:
            logger.info(f"Buscando en categoría: {category}")
            category_results = vectorstores[category].max_marginal_relevance_search(
                query=query,
                k=docs_per_category,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )
            
            logger.info(f"Encontrados {len(category_results)} documentos en '{category}'")
            
            for doc in category_results:
                if 'category' not in doc.metadata:
                    doc.metadata['category'] = category
            
            results.extend(category_results)
            
        except Exception as e:
            logger.error(f"Error al buscar en {category}: {e}")
            continue

    logger.info(f"Total de documentos recuperados: {len(results)}")
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
        if not doc.page_content or not doc.page_content.strip():
            logger.warning(f"Documento vacío detectado, ID: {doc.metadata.get('doc_id', 'unknown')}")
            continue
            
        if current_length + len(doc.page_content) <= max_length:
            context_parts.append(doc.page_content)
            current_length += len(doc.page_content)
            
            source = doc.metadata.get('source', '')
            if not source:
                source = doc.metadata.get('file_path', '')
            
            if source and source not in sources:
                sources.append(source)

    if not context_parts:
        logger.warning("No se pudo extraer contexto de los documentos recuperados")
        return "", sources
        
    return "\n\n".join(context_parts), sources

def retrieve_by_category(
    vectorstores: Dict[str, Chroma],
    category: str,
    query: str,
    k: int = 5,
    fetch_k_multiplier: int = 2,
    lambda_mult: float = 0.7
) -> List[Document]:
    """
    Función genérica para recuperar documentos de una categoría específica con manejo mejorado de errores.
    """
    if category not in vectorstores:
        logger.warning(f"Categoría '{category}' no encontrada en los vectorstores disponibles")
        logger.error(f"❌ Categoría '{category}' no encontrada. Categorías disponibles: {list(vectorstores.keys())}")
        return []
        
    try:
        fetch_k = k * fetch_k_multiplier
        logger.info(f"Buscando en '{category}', k={k}, fetch_k={fetch_k}, lambda_mult={lambda_mult}")
        
        documents = vectorstores[category].max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )
        
        logger.info(f"Recuperados {len(documents)} documentos de '{category}'")
        logger.info(f"✅ Se han recuperado {len(documents)} documentos de la categoría '{category}'")
        
        for doc in documents:
            if 'category' not in doc.metadata:
                doc.metadata['category'] = category
                
        return documents
        
    except Exception as e:
        logger.error(f"Error al recuperar documentos de '{category}': {e}")
        logger.error(f"❌ Error al recuperar documentos de la categoría {category}: {e}")
        return []

def create_rag_chain(llm_model):
    """
    Crea una cadena RAG usando LangChain con instrucciones específicas para el entorno docente.
    """
    # Nuevo template con instrucciones actualizadas para agentes especializados en tareas docentes
    template = """
Eres un agente especializado en tareas docentes, encargado de ayudar en evaluaciones, planificaciones o guías de estudio, según lo indique el agente supervisor (router_agent). 
Utiliza únicamente la información proporcionada en el contexto, el cual proviene de documentos oficiales (como bases curriculares, orientaciones, actividades sugeridas y propuestas pedagógicas).
Si no tienes suficiente información, responde: "No tengo suficiente información para responder a esta pregunta".

IMPORTANTE: Al final de tu respuesta, incluye una breve sección que cite las fuentes consultadas, usando el formato: "Fuentes: [fuente1, fuente2, ...]".

Contexto:
{context}

Pregunta:
{question}

Respuesta:
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm_model
        | StrOutputParser()
    )
    
    return rag_chain

def retrieve_and_generate(
    vectorstores: Dict[str, Chroma],
    query: str,
    categories: Optional[List[str]] = None,
    k: int = 5,
    llm_model: Any = None
) -> Dict[str, Any]:
    """
    Realiza la recuperación de documentos y genera una respuesta usando RAG.
    """
    docs = retrieve_documents(vectorstores, query, categories, k)
    context, sources = get_context_from_documents(docs)
    
    if not context:
        return {
            "response": "No he encontrado información relevante para tu pregunta.",
            "sources": [],
            "documents": []
        }
    
    rag_chain = create_rag_chain(llm_model)
    response = rag_chain.invoke({"context": context, "question": query})
    
    return {
        "response": response,
        "sources": sources,
        "documents": docs
    }

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
