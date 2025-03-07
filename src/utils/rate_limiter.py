import time
from ratelimit import limits, sleep_and_retry
from typing import Callable, Any

# Definir el límite de peticiones
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