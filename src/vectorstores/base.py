from abc import ABC, abstractmethod

import numpy as np


class VectorStore(ABC):
    """Interface for embedding providers."""

    @property
    @abstractmethod
    def build(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def save(self) -> np.ndarray:
        raise NotImplementedError

    
