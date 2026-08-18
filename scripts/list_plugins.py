from src.plugins.registry_loader import load_all_plugins
from src.plugins.registry import chunker_registry, embedding_registry, vector_store_registry, retriever_registry

def list_all_plugins():

    load_all_plugins()

    print("Chunkers: ", chunker_registry.names())
    print("Embeddings:", embedding_registry.names())
    print("Vector stores:", vector_store_registry.names())
    print("Retrivals: ", retriever_registry.names())


if __name__ == "__main__":
    list_all_plugins()
