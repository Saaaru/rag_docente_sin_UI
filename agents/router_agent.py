import datetime
import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from utils import rate_limited_llm_call

def create_router_agent(llm, planning_agent, evaluation_agent, study_guide_agent):
    """
    Crea un agente router que identifica el tipo de solicitud, verifica información
    completa y solo cuando tiene todos los datos necesarios deriva al especialista.
    """
    system_prompt = """Eres un agente router inteligente que analiza solicitudes educativas.
    
    Tu función es triple:
    
    1. IDENTIFICAR el tipo de contenido educativo solicitado:
       - PLANIFICACION (planes de clase, anuales, unidades, etc.)
       - EVALUACION (pruebas, exámenes, rúbricas, etc.)
       - GUIA (guías de estudio, material de repaso, fichas, etc.)
    
    2. VERIFICAR que la solicitud contenga estos datos ESENCIALES:
       - ASIGNATURA (Lenguaje, Matemáticas, Historia, Ciencias, etc.)
       - NIVEL EDUCATIVO (1° básico, 5° básico, 2° medio, etc.)
       - MES DEL AÑO (opcional, pero importante para la progresión)
    
    3. DETERMINAR si falta información para procesar la solicitud
    
    INFORMACIÓN IMPORTANTE:
    - Los niveles educativos van desde 1° básico (6 años) hasta 4° medio (18 años)
    - El año escolar chileno va de marzo a diciembre
    - La dificultad debe adaptarse al nivel educativo y mes del año
    - Si mencionan varios meses, el contenido posterior debe ser más avanzado
    
    Responde SOLO en formato JSON con esta estructura:
    {
        "tipo": "PLANIFICACION|EVALUACION|GUIA",
        "asignatura": "nombre de la asignatura o null",
        "nivel": "nivel educativo o null",
        "mes": "mes o meses mencionados o null",
        "informacion_completa": true/false,
        "informacion_faltante": ["asignatura", "nivel"] o [] si no falta nada
    }
    
    IMPORTANTE: Usa comillas dobles para las cadenas y null para valores nulos.
    """
    
    def router_execute(query, asignatura=None, nivel=None, mes=None):
        """
        Función del router que analiza la consulta, solicita información faltante
        y deriva al especialista adecuado cuando tiene todos los datos.
        
        Args:
            query: Consulta del usuario
            asignatura: Asignatura previamente identificada (opcional)
            nivel: Nivel educativo previamente identificado (opcional)
            mes: Mes o meses del año escolar (opcional)
            
        Returns:
            Tupla con (respuesta, necesita_info, info_actual, tipo_contenido)
        """
        # Si no se ha especificado un mes, usamos el mes actual, pero no lo consideramos obligatorio
        if not mes:
            meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", 
                     "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
            mes_actual = datetime.datetime.now().month
            mes = meses[mes_actual - 1]
            print(f"\n📅 Usando mes actual para router: {mes}")
            
        # Definimos una función auxiliar para manejar errores y simplificar el código
        def solicitar_info_faltante():
            response = "Para ayudarte mejor, necesito la siguiente información:\n\n"
            if not asignatura:
                response += "- ¿Para qué asignatura necesitas el material? (Ej: Matemáticas, Lenguaje, etc.)\n"
            if not nivel:
                response += "- ¿Para qué nivel educativo? (Ej: 2° básico, 8° básico, 3° medio, etc.)\n"
            return response, True, {"asignatura": asignatura, "nivel": nivel, "mes": mes}, "PENDIENTE"
            
        # Función auxiliar para invocar agente y manejar errores
        def invocar_agente_especializado(tipo):
            try:
                if tipo == "PLANIFICACION":
                    response, _, _ = planning_agent(query, asignatura, nivel, mes)
                    return response, False, {"asignatura": asignatura, "nivel": nivel, "mes": mes}, "PLANIFICACION"
                elif tipo == "EVALUACION":
                    response, _, _ = evaluation_agent(query, asignatura, nivel, mes)
                    return response, False, {"asignatura": asignatura, "nivel": nivel, "mes": mes}, "EVALUACION"
                else:  # GUIA
                    response, _, _ = study_guide_agent(query, asignatura, nivel, mes)
                    return response, False, {"asignatura": asignatura, "nivel": nivel, "mes": mes}, "GUIA"
            except Exception as e:
                print(f"\n⚠️ Error al invocar agente especializado: {e}")
                error_msg = "Lo siento, ha ocurrido un error al procesar tu solicitud. Por favor, intenta nuevamente con una consulta más clara."
                return error_msg, False, {"asignatura": asignatura, "nivel": nivel, "mes": mes}, tipo
        
        # Si ya tenemos asignatura y nivel, podemos determinar el tipo de contenido y derivar directamente
        if asignatura and nivel:
            # Determinar tipo de contenido
            try:
                prompt = [
                    SystemMessage(content="Identifica qué tipo de contenido educativo solicita el usuario: PLANIFICACION, EVALUACION o GUIA. Responde solo con una de estas palabras."),
                    HumanMessage(content=query)
                ]
                result = rate_limited_llm_call(llm.invoke, prompt)
                tipo = result.content.strip().upper()
                
                # Normalizar el tipo
                if "PLAN" in tipo:
                    tipo = "PLANIFICACION"
                elif "EVAL" in tipo:
                    tipo = "EVALUACION"
                elif "GU" in tipo:
                    tipo = "GUIA"
                else:
                    tipo = "PLANIFICACION"  # Por defecto
                
                # Derivar al especialista
                return invocar_agente_especializado(tipo)
                
            except Exception as e:
                print(f"\n⚠️ Error al determinar tipo de contenido: {e}")
                # En caso de error, usar planificación por defecto
                return invocar_agente_especializado("PLANIFICACION")
        
        # Si no tenemos toda la información, analizamos la consulta
        try:
            # Obtener la clasificación del LLM
            prompt = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
            
            result = rate_limited_llm_call(llm.invoke, prompt)
            
            # Parsear el resultado JSON de forma segura            
            # Extraer el bloque JSON de la respuesta
            json_str = result.content
            # Encontrar el primer '{' y el último '}'
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            
            if start != -1 and end != 0:
                json_str = json_str[start:end]
                
                # Limpiar y normalizar el JSON
                json_str = json_str.replace("'", '"')
                # Reemplazar valores de texto null por null de JSON
                json_str = re.sub(r'"null"', 'null', json_str)
                
                try:
                    # Parsear el JSON limpio
                    decision = json.loads(json_str)
                    
                    # Extraer información
                    tipo = decision.get("tipo", "").upper()
                    # Usar los valores pasados por parámetro si están disponibles
                    asignatura = asignatura or decision.get("asignatura")
                    nivel = nivel or decision.get("nivel")
                    mes = mes or decision.get("mes")
                    
                    # Convertir "null" a None
                    if asignatura == "null" or asignatura == "NULL":
                        asignatura = None
                    if nivel == "null" or nivel == "NULL":
                        nivel = None
                    if mes == "null" or mes == "NULL":
                        mes = None
                    
                    # Verificar si necesitamos más información
                    informacion_completa = decision.get("informacion_completa", False)
                    
                    # Si falta información, solicitarla
                    if not informacion_completa or not asignatura or not nivel:
                        return solicitar_info_faltante()
                    
                    # Si tenemos toda la información, derivar al especialista
                    return invocar_agente_especializado(tipo)
                    
                except json.JSONDecodeError as je:
                    print(f"\n⚠️ Error al decodificar JSON: {je}")
                    print(f"JSON problemático: {json_str}")
            
            # Si no se pudo parsear JSON o no se encontró JSON en la respuesta
            # Enfoque alternativo simplificado
            print("\n⚠️ Usando enfoque simplificado para determinar tipo y requisitos.")
            
            # Intentar determinar tipo directamente
            tipo = "PLANIFICACION"  # Valor predeterminado
            if "PLANIFICACIÓN" in query.upper() or "PLANIFICACION" in query.upper() or "PLAN" in query.upper():
                tipo = "PLANIFICACION"
            elif "EVALUACIÓN" in query.upper() or "EVALUACION" in query.upper() or "EVAL" in query.upper():
                tipo = "EVALUACION"
            elif "GUÍA" in query.upper() or "GUIA" in query.upper():
                tipo = "GUIA"
            
            # Verificar si tenemos información completa
            if asignatura and nivel:
                return invocar_agente_especializado(tipo)
            else:
                return solicitar_info_faltante()
                
        except Exception as e:
            print(f"\n⚠️ Error general en el router: {e}")
            
            # Intento de recuperación básico
            if asignatura and nivel:
                # Si tenemos información básica, intentar con planificación
                return invocar_agente_especializado("PLANIFICACION")
            else:
                # Si falta información básica, solicitarla
                return solicitar_info_faltante()
    
    return router_execute