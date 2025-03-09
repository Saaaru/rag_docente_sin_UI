import logging
import os
from datetime import datetime
from pathlib import Path
import time
from queue import Queue
import threading

class BufferedFileHandler(logging.FileHandler):
    """Handler personalizado que acumula registros y los escribe periódicamente"""
    def __init__(self, filename, mode='a', encoding=None, delay=False, flush_interval=10):
        super().__init__(filename, mode, encoding, delay)
        self.buffer = Queue()
        self.flush_interval = flush_interval
        self.last_flush = time.time()
        self.buffer_lock = threading.Lock()
        
        # Iniciar thread para flush periódico
        self.flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
        self.flush_thread.start()

    def emit(self, record):
        """Acumula los registros en el buffer en lugar de escribirlos inmediatamente"""
        try:
            msg = self.format(record)
            self.buffer.put(msg)
            
            # Si han pasado más de flush_interval segundos, hacer flush
            if time.time() - self.last_flush >= self.flush_interval:
                self.flush()
        except Exception:
            self.handleError(record)

    def flush(self):
        """Escribe todos los registros acumulados en el archivo"""
        with self.buffer_lock:
            try:
                while not self.buffer.empty():
                    msg = self.buffer.get()
                    self.stream.write(msg + '\n')
                self.stream.flush()
                self.last_flush = time.time()
            except Exception as e:
                print(f"Error al hacer flush del buffer: {e}")

    def _periodic_flush(self):
        """Ejecuta flush periódicamente"""
        while True:
            time.sleep(self.flush_interval)
            self.flush()

    def close(self):
        """Asegura que todos los registros se escriban antes de cerrar"""
        self.flush()
        super().close()

def setup_logger():
    """
    Configura y retorna un logger personalizado que escribe tanto en consola como en archivo.
    El archivo de log se guarda en la carpeta raíz del proyecto.
    """
    try:
        # Obtener el directorio raíz del proyecto (2 niveles arriba desde utils)
        root_dir = Path(__file__).parent.parent.parent
        
        # Asegurarse de que el directorio raíz existe
        root_dir.mkdir(exist_ok=True)
        
        # Crear nombre de archivo con timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = root_dir / f'log_{timestamp}.txt'
        
        print(f"Intentando crear archivo de log en: {log_file}")
        
        # Verificar si podemos escribir en el directorio
        if not os.access(root_dir, os.W_OK):
            print(f"¡Advertencia! No hay permisos de escritura en: {root_dir}")
            log_file = Path(f'./log_{timestamp}.txt')
        
        # Configurar el logger
        logger = logging.getLogger('EDU_AGENT')
        logger.setLevel(logging.DEBUG)
        
        # Limpiar handlers existentes
        if logger.handlers:
            logger.handlers.clear()
        
        # Formato detallado para el archivo
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s:\n'
            'Archivo: %(filename)s, Línea: %(lineno)d\n'
            'Mensaje: %(message)s\n'
            '-' * 80 + '\n',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Usar el nuevo BufferedFileHandler en lugar del FileHandler estándar
        try:
            file_handler = BufferedFileHandler(
                str(log_file), 
                encoding='utf-8', 
                mode='w',
                flush_interval=10  # Flush cada 10 segundos
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            print(f"Archivo de log creado exitosamente en: {log_file}")
        except Exception as e:
            print(f"Error al crear el archivo de log: {e}")
            log_file = Path(f'./log_{timestamp}.txt')
            file_handler = BufferedFileHandler(
                str(log_file), 
                encoding='utf-8', 
                mode='w',
                flush_interval=10
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            print(f"Archivo de log creado en ubicación alternativa: {log_file}")
        
        # Handler para consola (mantener los emojis y formato original)
        console_formatter = logging.Formatter('%(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Registrar inicio del log
        logger.info(f"=== INICIO DE SESIÓN ===")
        logger.info(f"Archivo de log creado en: {log_file}")
        logger.info("=" * 50)
        
        return logger
    
    except Exception as e:
        print(f"Error crítico al configurar el logger: {e}")
        # Configuración mínima de respaldo
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger('EDU_AGENT') 