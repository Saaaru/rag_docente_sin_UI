# RAG Docente - Asistente para Planificación Educativa

## Descripción
RAG Docente es una aplicación de Recuperación Aumentada por Generación (RAG) diseñada para ayudar a los docentes chilenos en la creación de planificaciones educativas basadas en el currículum nacional. El sistema utiliza modelos de IA avanzados para analizar documentos educativos en formato PDF y proporcionar respuestas específicas y contextualizadas a consultas relacionadas con planificaciones, evaluaciones y actividades educativas.

## Características principales
- **Recuperación inteligente de documentos**: Utiliza embeddings y vectores semánticos para encontrar información relevante en documentos del currículum chileno.
- **Respuestas contextualizadas**: Mantiene el contexto de la conversación para proporcionar respuestas coherentes.
- **Historial de conversaciones**: Guarda automáticamente las conversaciones para referencias futuras.
- **Optimización de recursos**: Evita recrear la base de datos vectorial si ya existe, ahorrando tiempo y recursos computacionales.

## Uso de la aplicación
El sistema está diseñado para asistir a los docentes en:
- Creación de planificaciones anuales y mensuales
- Desarrollo de actividades alineadas con el currículum
- Diseño de evaluaciones según objetivos de aprendizaje
- Consultas sobre marcos curriculares por nivel y asignatura

## Requisitos previos
- Python 3.8 o superior
- Cuenta en Google Cloud con API Vertex AI habilitada
- Archivo de credenciales de Google Cloud (JSON)
- Cuenta en LangSmith (opcional, para seguimiento de interacciones)

## Dependencias principales
```
langchain-core
langgraph
langchain_google_vertexai
langchain_chroma
langchain_community
flask
python-dotenv
ratelimit
```

## Instalación

1. Clona este repositorio:
```bash
git clone https://github.com/tu-usuario/rag_docente.git
cd rag_docente
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Configura las credenciales:
   - Coloca tu archivo de credenciales de Google Cloud en la carpeta `db/`
   - Crea un archivo `.env` en la carpeta `db/` con tu clave de API de LangSmith:
   ```
   LANGSMITH_API_KEY=tu_clave_aquí
   ```

4. Coloca tus documentos PDF:
   - Añade los documentos del currículum chileno en la carpeta `pdf_docs/`

## Uso

1. Ejecuta la aplicación:
```bash
python app.py
```

2. Interactúa con el asistente haciendo preguntas sobre el currículum o solicitando planificaciones.

3. Para salir, escribe 'exit', 'quit' o 'q'.

## Estructura del proyecto
- `app.py`: Archivo principal que contiene toda la lógica del sistema RAG
- `pdf_docs/`: Directorio donde se almacenan los documentos PDF
- `conversaciones/`: Directorio donde se guardan los registros de las conversaciones
- `pdf-rag-chroma/`: Directorio donde se almacena la base de datos vectorial
- `requirements.txt`: Lista de dependencias del proyecto

## Funcionamiento técnico

### Carga y procesamiento de documentos
Los documentos PDF se cargan y dividen en fragmentos más pequeños para su procesamiento. El tamaño de los fragmentos está optimizado para recuperar información relevante sin perder contexto.

### Vectorización y almacenamiento
Los fragmentos de texto se transforman en vectores numéricos (embeddings) usando el modelo `text-embedding-004` de Google Vertex AI. Estos vectores se almacenan en una base de datos Chroma para búsquedas rápidas.

### Sistema de recuperación
La aplicación utiliza un enfoque de recuperación mejorada mediante:
- **Enhanced Retriever**: Para consultas específicas que requieren respuestas precisas
- **Contextual Retriever**: Para consultas que necesitan una comprensión más amplia del contexto curricular

### Integración con Google Vertex AI
El sistema utiliza el modelo Gemini 1.5 Flash a través de la API de Vertex AI para generar respuestas coherentes y precisas basadas en la información recuperada.

### Gestión de memoria y estados
La aplicación implementa un grafo de estado (LangGraph) que mantiene el historial de conversación y proporciona respuestas contextualizadas.

## Mejoras y optimizaciones
- La aplicación verifica si ya existe una base de datos vectorial antes de crearla, ahorrando tiempo en inicios posteriores.
- Implementa limitación de tasa (rate limiting) para evitar problemas con APIs externas.
- Utiliza procesamiento asíncrono para mejorar la experiencia del usuario.

## Notas
- El rendimiento del sistema depende de la calidad y cantidad de los documentos PDF proporcionados.
- Se recomienda utilizar documentos oficiales del Ministerio de Educación de Chile para obtener los mejores resultados.

## Licencia
Este proyecto está bajo la licencia MIT. 

---

Desarrollado para asistir a la comunidad educativa chilena en la creación de planificaciones alineadas con el currículum nacional. 