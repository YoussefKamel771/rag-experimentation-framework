from __future__ import annotations
from typing import Any, Callable, TypeVar

T=TypeVar("T")

class PluginRegistry:
    def __init__(self, name: str):
        self.name = name
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, plugin_name: str):
        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if plugin_name in self._factories:
                raise ValueError(f"Plugin '{plugin_name}' already registered in {self.name}")
            self._factories[plugin_name]=factory
            return factory
        return decorator 
    
    def create(self, plugin_name: str, **kwargs: Any):
        if plugin_name not in self._factories:
            raise ValueError(f"Unknown {self.name} plugin '{plugin_name}'. Available: {self.names()}")
        return self._factories[plugin_name](**kwargs)
    
    def names(self):
        return sorted(self._factories)

chunker_registry=PluginRegistry("chunker")
embedding_registry=PluginRegistry("embedding")
vector_store_registry=PluginRegistry("vector_store")
