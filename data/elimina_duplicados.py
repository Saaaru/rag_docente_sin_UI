import os
import filecmp
import shutil
import argparse
from collections import defaultdict
from tqdm import tqdm

def encontrar_duplicados_simple(directorio_raiz):
    """Encuentra archivos duplicados basados en tamaño y comparación directa"""
    # Paso 1: Agrupar archivos por tamaño
    print(f"Paso 1: Agrupando archivos por tamaño en {directorio_raiz}...")
    archivos_por_tamano = defaultdict(list)
    
    archivos_totales = sum([len(files) for _, _, files in os.walk(directorio_raiz)])
    with tqdm(total=archivos_totales, desc="Analizando archivos") as pbar:
        for carpeta_actual, _, archivos in os.walk(directorio_raiz):
            for nombre_archivo in archivos:
                ruta_completa = os.path.join(carpeta_actual, nombre_archivo)
                try:
                    tamano = os.path.getsize(ruta_completa)
                    # Solo considerar archivos no vacíos
                    if tamano > 0:
                        archivos_por_tamano[tamano].append(ruta_completa)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error al procesar {ruta_completa}: {e}")
                    pbar.update(1)
    
    # Paso 2: Para archivos del mismo tamaño, comparar contenido directamente
    print("\nPaso 2: Identificando duplicados mediante comparación directa...")
    grupos_duplicados = []
    
    # Filtrar grupos con más de un archivo
    grupos_potenciales = {tam: rutas for tam, rutas in archivos_por_tamano.items() if len(rutas) > 1}
    print(f"  Encontrados {len(grupos_potenciales)} grupos de archivos con tamaño idéntico.")
    
    # Debug: mostrar algunos ejemplos de grupos
    for tamano, rutas in sorted(list(grupos_potenciales.items())[:5]):
        print(f"Ejemplo - Tamaño {tamano} bytes: {len(rutas)} archivos")
        print(f"  Nombres: {[os.path.basename(r) for r in rutas[:min(3, len(rutas))]]}")
    
    for tamano, rutas in tqdm(grupos_potenciales.items(), desc="Verificando duplicados"):
        archivos_procesados = set()
        
        for i, ruta1 in enumerate(rutas):
            if ruta1 in archivos_procesados:
                continue
                
            duplicados = [ruta1]
            
            for j in range(i+1, len(rutas)):
                ruta2 = rutas[j]
                if ruta2 in archivos_procesados:
                    continue
                
                # Usar comparación directa de archivos
                try:
                    if filecmp.cmp(ruta1, ruta2, shallow=False):
                        duplicados.append(ruta2)
                        archivos_procesados.add(ruta2)
                except Exception as e:
                    print(f"Error al comparar {ruta1} y {ruta2}: {e}")
            
            if len(duplicados) > 1:
                grupos_duplicados.append(duplicados)
                archivos_procesados.add(ruta1)
    
    print(f"\nSe encontraron {len(grupos_duplicados)} grupos de archivos duplicados.")
    if grupos_duplicados:
        print("Ejemplos de duplicados encontrados:")
        for i, grupo in enumerate(grupos_duplicados[:3], 1):
            print(f"Grupo {i}: {[os.path.basename(g) for g in grupo]}")
    
    return grupos_duplicados

def eliminar_duplicados(grupos_duplicados, conservar_mas_reciente=True, hacer_backup=False, 
                       dir_backup=None, directorio_raiz=None):
    """Elimina los archivos duplicados"""
    if not grupos_duplicados:
        print("No hay duplicados para eliminar.")
        return 0, 0
        
    archivos_eliminados = 0
    bytes_liberados = 0
    
    # Crear directorio de backup si es necesario
    if hacer_backup:
        if dir_backup is None and directorio_raiz is not None:
            dir_backup = os.path.join(os.path.dirname(directorio_raiz), "_backup_duplicados")
        
        if dir_backup and not os.path.exists(dir_backup):
            os.makedirs(dir_backup)
            print(f"Creado directorio de backup: {dir_backup}")
    
    print("\nProcesando grupos de archivos duplicados:")
    
    for i, grupo in enumerate(grupos_duplicados, 1):
        # Determinar qué archivo conservar
        if conservar_mas_reciente:
            # Ordenar por tiempo de modificación (más reciente primero)
            grupo.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        else:
            # Ordenar por tiempo de creación (más antiguo primero)
            grupo.sort(key=lambda x: os.path.getctime(x))
        
        # El primer archivo se conserva, el resto se elimina
        archivo_conservado = grupo[0]
        archivos_a_eliminar = grupo[1:]
        
        print(f"\nGrupo {i}/{len(grupos_duplicados)} ({len(grupo)} archivos):")
        print(f"  Conservando: {archivo_conservado}")
        
        for ruta in archivos_a_eliminar:
            try:
                tamano = os.path.getsize(ruta)
                
                # Hacer backup si es necesario
                if hacer_backup and dir_backup:
                    nombre_backup = os.path.basename(ruta)
                    # Añadir un número al nombre si ya existe
                    contador = 1
                    nombre_final = nombre_backup
                    while os.path.exists(os.path.join(dir_backup, nombre_final)):
                        nombre_base, extension = os.path.splitext(nombre_backup)
                        nombre_final = f"{nombre_base}_{contador}{extension}"
                        contador += 1
                    
                    shutil.copy2(ruta, os.path.join(dir_backup, nombre_final))
                    print(f"  Backup creado: {nombre_final}")
                
                # Eliminar el archivo
                os.remove(ruta)
                print(f"  Eliminado: {ruta} ({tamano} bytes)")
                
                archivos_eliminados += 1
                bytes_liberados += tamano
            except Exception as e:
                print(f"  Error al eliminar {ruta}: {e}")
    
    # Mostrar estadísticas
    mb_liberados = bytes_liberados / (1024 * 1024)
    
    print(f"\nEliminación completada. Se eliminaron {archivos_eliminados} archivos.")
    print(f"Se liberaron aproximadamente {mb_liberados:.2f} MB.")
    
    return archivos_eliminados, bytes_liberados

def main():
    # Configuración del directorio raíz
    DIRECTORIO_RAIZ = "C:/Users/Dante/Desktop/rag_docente_sin_UI-2/src/data/raw"  # Puedes cambiar esto por tu ruta
    
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Eliminar archivos duplicados')
    parser.add_argument('--directorio', default=DIRECTORIO_RAIZ,
                      help=f'Directorio a analizar (default: {DIRECTORIO_RAIZ})')
    parser.add_argument('--conservar-antiguo', action='store_true',
                      help='Conservar el archivo más antiguo en lugar del más reciente')
    parser.add_argument('--backup', action='store_true',
                      help='Hacer backup de los archivos antes de eliminarlos')
    parser.add_argument('--dir-backup',
                      help='Directorio donde guardar los backups (por defecto: _backup_duplicados)')
    
    args = parser.parse_args()
    
    print("\n=== BUSCADOR DE ARCHIVOS DUPLICADOS (VERSIÓN MEJORADA) ===")
    print(f"Directorio a analizar: {args.directorio}")
    
    # Asegurarse de que tqdm está instalado
    try:
        import tqdm
    except ImportError:
        print("Instalando dependencias necesarias...")
        import pip
        pip.main(['install', 'tqdm'])
    
    # Encontrar duplicados con el método simplificado
    grupos_duplicados = encontrar_duplicados_simple(args.directorio)
    
    # Eliminar duplicados si se encontraron
    if grupos_duplicados:
        print("\n¡ATENCIÓN! Se procederá a eliminar archivos duplicados.")
        print("Por favor, asegúrate de tener respaldos importantes antes de continuar.")
        respuesta = input("¿Deseas continuar con la eliminación? (s/n): ")
        
        if respuesta.lower() == 's':
            eliminar_duplicados(
                grupos_duplicados, 
                not args.conservar_antiguo,  # Por defecto conserva el más reciente
                args.backup,
                args.dir_backup,
                args.directorio
            )
        else:
            print("Operación cancelada por el usuario.")
    else:
        print("No se encontraron archivos duplicados.")

if __name__ == "__main__":
    main()