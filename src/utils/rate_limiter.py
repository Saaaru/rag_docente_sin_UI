import time
from functools import wraps
from typing import Callable, Any

# Definir el límite de peticiones
CALLS_PER_MINUTE = 150
PERIOD = 60
WAIT_TIME = 1

def rate_limited_llm_call(method: Callable) -> Callable:
    """
    Decorador para métodos de clase con rate limiting
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs) -> Any:
        time.sleep(WAIT_TIME)
        return method(self, *args, **kwargs)
    return wrapper