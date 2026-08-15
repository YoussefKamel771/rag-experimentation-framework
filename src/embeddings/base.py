from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    """Interface for embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Return a float32 matrix of shape:
            (number_of_texts, embedding_dimension)
        """
        raise NotImplementedError
