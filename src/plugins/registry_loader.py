def load_all_plugins():
    import src.chunking.recursive
    import src.embeddings.sentence_transformers
    import src.embeddings.ollama
    import src.vectorstores.faiss_store
    import src.vectorstores.qdrant_store