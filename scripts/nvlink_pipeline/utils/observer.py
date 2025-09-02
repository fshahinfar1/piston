from typing import *

class Observable:
    def __init__(self):
        self._events: Dict[str, List[Tuple[Callable, Any]]] = {}
    
    def register(self, key: str, callback: Callable, user_state=None):
        callbacks = self._events.setdefault(key, [])
        callbacks.append((callback, user_state))
    
    def _notify(self, key, *args, **kwargs):
        callbacks, user_state = self._events.get(key)
        if callbacks is None:
            return
        for fn in callbacks:
            fn(user_state, *args, **kwargs)
