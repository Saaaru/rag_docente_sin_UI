from .loader import (
    initialize_vectorstore,
    COLLECTION_NAMES,
    EmbeddingsManager,  #  Aunque no se use, se necesita para la creación
)
from .retriever import (
    retrieve_documents,
    retrieve_with_filter,
    get_context_from_documents,
    retrieve_by_category,
    retrieve_bases_curriculares,
    retrieve_actividades_sugeridas,
    retrieve_orientaciones,
    retrieve_propuesta
)

__all__ = [
    'initialize_vectorstore',
    'COLLECTION_NAMES',
    'EmbeddingsManager',
    'retrieve_documents',
    'retrieve_with_filter',
    'get_context_from_documents',
    'retrieve_by_category',
    'retrieve_bases_curriculares',
    'retrieve_actividades_sugeridas',
    'retrieve_orientaciones',
    'retrieve_propuesta'
]