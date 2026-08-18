from enum import Enum

class VectorDBEnums(Enum):
    QDRANT = "qdrant"
    FAISS = "faiss"

class DistanceMethodEnums(Enum):
    COSINE = "cosine"
    DOT = "dot"
    