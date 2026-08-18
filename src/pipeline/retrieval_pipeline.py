from __future__ import annotations
from typing import Any
from src.plugins.registry import retriever_registry
from src.plugins.registry_loader import load_all_plugins
from configs.config import Settings

from src.models.enums.VectorDBEnums import VectorDBEnums
from src.models.enums.RetrivalTypeEnums import RetrivalTypeEnums

def create_dense_retriever(config: Settings):

    source = config.vector_store
    embedding = config.embedding

    if source.type == VectorDBEnums.FAISS.value:
        return retriever_registry.create(
            RetrivalTypeEnums.DENSE_FAISS.value,
            index_dir=source.params["path"],

            embedding_provider=(
                embedding.provider
            ),

            embedding_params=(
                embedding.params
            ),
        )

    if source.type == VectorDBEnums.QDRANT.value:
        return retriever_registry.create(
            RetrivalTypeEnums.DENSE_QDRANT.value,
            path = source.params["path"],
            
            collection_name=(
                source.params["collection_name"]
            ),

            embedding_provider=(
                embedding.provider
            ),

            embedding_params=(
                embedding.params
            ),
        )
    raise ValueError(
        f"Unsupported dense source: "
        f"{source.type}"
    )

def create_retriever(config: Settings):
    load_all_plugins()

    retrieval = config.retrieval

    if retrieval.type == RetrivalTypeEnums.DENSE.value:
        return create_dense_retriever(config)

    if retrieval.type ==  RetrivalTypeEnums.LEXICAL.value:
        if retrieval.lexical is None:
            raise ValueError(
                "Lexical configuration is required "
                "for BM25 retrieval"
            )
        return retriever_registry.create(
            retrieval.lexical.type,
            **retrieval.lexical.params,
        )

    if retrieval.type == RetrivalTypeEnums.HYBRID.value:

        if retrieval.dense is None:
            raise ValueError(
                "Dense configuration is required "
                "for hybrid retrieval"
            )

        if retrieval.lexical is None:
            raise ValueError(
                "Lexical configuration is required "
                "for hybrid retrieval"
            )

        dense_retriever = create_dense_retriever(config)

        lexical_retriever = retriever_registry.create(
                        retrieval.lexical.type,
                        **retrieval.lexical.params,
                    )
            

        return retriever_registry.create(
            RetrivalTypeEnums.HYBRID.value,
            dense_retriever=dense_retriever,
            lexical_retriever=lexical_retriever,
            **retrieval.params,
        )

    raise ValueError(
        f"Unknown retriever type: "
        f"{retrieval.type}"
    )

    

def retrieve(
    config: Settings,
    query: str,
):

    retriever = create_retriever(config)

    return retriever.retrieve(
        query,
        top_k=config.retrieval.top_k,
    )
