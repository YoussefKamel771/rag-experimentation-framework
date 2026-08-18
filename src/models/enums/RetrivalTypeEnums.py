from enum import Enum

class RetrivalTypeEnums(Enum):
    DENSE = "dense"
    LEXICAL = "bm25"
    HYBRID = "hybrid"

    DENSE_QDRANT = "dense_qdrant"
    DENSE_FAISS = "dense_faiss"
    