from typing import *

class Observable:
    def __init__(self):
        self._events: Dict[str, List[Tuple[Callable, Any]]] = {}
    
    def register(self, key: str, callback: Callable, user_state=None) -> None:
        callbacks = self._events.setdefault(key, [])
        callbacks.append((callback, user_state))
    
    def unregister(self, key: str, callback: Callable) -> None:
        self._events[key] = [t for t in self._events[key] if t[0] != callback]
    
    def _notify(self, key, *args, **kwargs) -> None:
        list_listeners = self._events.get(key)
        if not list_listeners:
            return

        for fn, user_state in list_listeners:
            fn(user_state, *args, **kwargs)
