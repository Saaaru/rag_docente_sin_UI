import os
import hashlib
from collections import defaultdict

def calcular_hash_archivo(ruta_archivo):
    """Calcula el hash SHA-256 de un archivo para identificarlo de forma única."""
    hash_sha256 = hashlib.sha256()
    with open(ruta_archivo, 'rb') as f:
        # Leer el archivo en bloques para manejar archivos grandes
        for bloque in iter(lambda: f.read(4096), b''):
            hash_sha256.update(bloque)
    return hash_sha256.hexdigest()

def encontrar_duplicados(directorio_raiz, extension='.pdf'):
    """
    Encuentra archivos duplicados en el directorio raíz y sus subdirectorios.
    
    Args:
        directorio_raiz: Directorio desde donde comenzar la búsqueda
        extension: Extensión de archivo a buscar (por defecto .pdf)
        
    Returns:
        Un diccionario donde las claves son los hashes y los valores son listas
        de rutas de archivos con contenido idéntico
    """
    # Diccionario para almacenar archivos por su hash (contenido)
    archivos_por_hash = defaultdict(list)
    
    # Recorrer todos los directorios y subdirectorios
    for directorio_actual, subdirectorios, archivos in os.walk(directorio_raiz):
        for archivo in archivos:
            if archivo.lower().endswith(extension.lower()):
                ruta_completa = os.path.join(directorio_actual, archivo)
                # Calcular el hash del archivo
                hash_archivo = calcular_hash_archivo(ruta_completa)
                # Añadir la ruta del archivo a la lista correspondiente a su hash
                archivos_por_hash[hash_archivo].append(ruta_completa)
    
    # Filtrar para obtener solo los grupos de archivos con duplicados
    duplicados = {hash_val: rutas for hash_val, rutas in archivos_por_hash.items() if len(rutas) > 1}
    
    return duplicados

def eliminar_duplicados(duplicados, modo_seguro=True):
    """
    Elimina archivos duplicados manteniendo uno de cada grupo.
    
    Args:
        duplicados: Diccionario donde las claves son hashes y los valores son 
                    listas de rutas de archivos con contenido idéntico
        modo_seguro: Si es True, solo muestra qué archivos se eliminarían sin eliminarlos
        
    Returns:
        Una lista de los archivos eliminados
    """
    archivos_eliminados = []
    
    for hash_val, rutas in duplicados.items():
        # Mantener el primer archivo y eliminar los demás
        archivo_a_mantener = rutas[0]
        archivos_a_eliminar = rutas[1:]
        
        print(f"Manteniendo: {archivo_a_mantener}")
        
        for ruta in archivos_a_eliminar:
            if modo_seguro:
                print(f"Se eliminaría (modo seguro): {ruta}")
            else:
                try:
                    os.remove(ruta)
                    print(f"Eliminado: {ruta}")
                    archivos_eliminados.append(ruta)
                except Exception as e:
                    print(f"Error al eliminar {ruta}: {e}")
    
    return archivos_eliminados

def main():
    # Configuración
    directorio_raiz = r"C:/Users/Dante/Desktop/rag_docente_sin_UI-1/data/raw/pdf_docs"
    extension = input("Introduce la extensión de archivo a buscar (por defecto .pdf): ") or '.pdf'
    
    # Validar que el directorio existe
    if not os.path.isdir(directorio_raiz):
        print(f"El directorio {directorio_raiz} no existe.")
        return
    
    print(f"Buscando archivos duplicados con extensión {extension} en {directorio_raiz}...")
    duplicados = encontrar_duplicados(directorio_raiz, extension)
    
    if not duplicados:
        print("No se encontraron archivos duplicados.")
        return
    
    # Mostrar estadísticas
    total_grupos = len(duplicados)
    total_duplicados = sum(len(rutas) - 1 for rutas in duplicados.values())
    
    print(f"\nSe encontraron {total_grupos} grupos de archivos duplicados.")
    print(f"Total de archivos duplicados a eliminar: {total_duplicados}")
    
    # Mostrar detalles de los duplicados
    for i, (hash_val, rutas) in enumerate(duplicados.items(), 1):
        print(f"\nGrupo {i} (Hash: {hash_val[:8]}...):")
        for ruta in rutas:
            print(f"  - {ruta}")
    
    # Preguntar si se desea eliminar los duplicados
    respuesta = input("\n¿Deseas eliminar los duplicados? (s/n): ").lower()
    
    if respuesta == 's':
        modo_seguro = input("¿Ejecutar en modo seguro (solo mostrar qué archivos se eliminarían)? (s/n): ").lower() == 's'
        eliminados = eliminar_duplicados(duplicados, modo_seguro)
        
        if modo_seguro:
            print(f"\nModo seguro: Se habrían eliminado {total_duplicados} archivos duplicados.")
        else:
            print(f"\nSe eliminaron {len(eliminados)} archivos duplicados.")
    else:
        print("Operación cancelada. No se eliminó ningún archivo.")

if __name__ == "__main__":
    main()