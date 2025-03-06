import os
import time
import datetime
import sys
from ratelimit import limits, sleep_and_retry
from typing import Callable, Any

# Configure stdout to handle special characters properly
sys.stdout.reconfigure(encoding='utf-8')

# Definir el límite de peticiones (por ejemplo, 180 por minuto para estar seguros)
CALLS_PER_MINUTE = 150
PERIOD = 60
WAIT_TIME = 1

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=PERIOD)
def rate_limited_llm_call(func: Callable, *args, **kwargs) -> Any:
    """
    Wrapper function para las llamadas al LLM con rate limiting mejorado
    """
    time.sleep(WAIT_TIME)
    return func(*args, **kwargs)

def format_and_save_conversation(query: str, response: str, thread_id: str, output_dir: str = "conversaciones") -> str:
    """
    Formatea y guarda la conversación en un archivo Markdown usando el thread_id y timestamp.
    """
    # Crear directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Crear timestamp en formato dd/mm/aaaa_HH:MM
    timestamp = time.strftime("%d-%m-%Y_%H-%M")
    
    # Usar el thread_id y timestamp en el nombre del archivo
    filename = f"conversacion_{timestamp}_ID_{thread_id}.md"
    filepath = os.path.join(output_dir, filename)

    # Si el archivo ya existe, añadir al contenido existente
    existing_content = ""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            existing_content = f.read()

    # Formatear el nuevo contenido
    new_content = f"""
## Pregunta
{query}

## Respuesta
{response}

---
"""

    # Combinar contenido existente con nuevo contenido
    markdown_content = existing_content + new_content if existing_content else f"""# Conversación RAG - Thread ID: {thread_id}
Iniciada el: {time.strftime("%d/%m/%Y %H:%M")}

{new_content}"""

    # Guardar el archivo
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"\n💾 Conversación guardada en: {filepath}")
    return filepath