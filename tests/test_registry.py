from src.plugins.registry_loader import load_all_plugins
from src.plugins.registry import chunker_registry, embedding_registry, vector_store_registry

def test_plugins():
    load_all_plugins()
    assert {"fixed","recursive","sentence","markdown","semantic"} <= set(chunker_registry.names())
    assert {"ollama","sentence_transformers"} <= set(embedding_registry.names())
    assert {"faiss","qdrant"} <= set(vector_store_registry.names())
