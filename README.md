# RAG Docente - Asistente para Planificación Educativa

## Descripción General
RAG Docente es una aplicación basada en la técnica de Recuperación Aumentada por Generación (RAG) que asiste a los docentes chilenos en la creación de planificaciones educativas alineadas con el currículo nacional. La aplicación combina el procesamiento de documentos en formato PDF, técnicas de vectorización mediante embeddings y modelos avanzados de inteligencia artificial para responder consultas específicas y contextualizadas.

## Características Principales
- **Recuperación Inteligente de Documentos:**  
  Utiliza técnicas de embeddings y vectores semánticos para extraer información relevante de documentos oficiales del currículo.
  
- **Respuestas Contextualizadas:**  
  Mantiene el contexto de la conversación mediante un sistema de historial (LangGraph), asegurando respuestas coherentes a lo largo del tiempo.
  
- **Optimización de Recursos:**  
  Verifica la existencia de la base de datos vectorial (almacenada en `pdf-rag-chroma/`) para evitar reprocesamientos innecesarios y mejorar la eficiencia.
  
- **Integración con Google Vertex AI:**  
  Aprovecha modelos como Gemini 1.5 Flash y `text-embedding-004` para generar respuestas precisas y transformar fragmentos de texto en vectores numéricos.

- **Escalabilidad y Modularidad:**  
  La arquitectura está diseñada para separar las responsabilidades (procesamiento de PDFs, vectorización, gestión del historial) y facilitar el mantenimiento y la ejecución de pruebas.

## Funcionamiento Técnico
### 1. Procesamiento de Documentos
- Los documentos PDF se almacenan en la carpeta `pdf_docs/`.
- Se dividen en fragmentos optimizados para preservar el contexto y facilitar su análisis.
- Es recomendable usar documentos oficiales del Ministerio de Educación de Chile para obtener resultados óptimos.

### 2. Vectorización y Almacenamiento
- Cada fragmento se transforma en un vector numérico utilizando el modelo `text-embedding-004` de Vertex AI.
- Los vectores resultantes se almacenan en una base de datos Chroma, permitiendo búsquedas semánticas rápidas.
- Se valida si ya existe una base de datos previa para evitar reprocesar los documentos.

### 3. Sistema de Recuperación y Generación de Respuestas
- **Enhanced Retriever:** Utilizado para consultas que requieren respuestas más puntuales y precisas.
- **Contextual Retriever:** Permite el mantenimiento del contexto en conversaciones extendidas.
- La generación de respuestas se realiza mediante el modelo Gemini 1.5 Flash a través de la API de Google Vertex AI.

### 4. Gestión de Estado y Memoria
- Se implementa un grafo de estados (LangGraph) que registra todo el historial de conversaciones, ayudando a mantener el contexto y a gestionar interacciones complejas.

## Instalación y Configuración
### Requisitos Previos
- Python 3.8 o superior.
- Cuenta en Google Cloud con la API de Vertex AI habilitada.
- Archivo de credenciales de Google Cloud (JSON) ubicado en la carpeta `db/`.
- (Opcional) Cuenta en LangSmith para seguimiento de interacciones.

### Dependencias Principales
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