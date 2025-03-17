# Ya no necesitamos importar de server.py
# from .server import app

# Importamos directamente de routes
from .routes import chat_router

__all__ = ["chat_router"]
