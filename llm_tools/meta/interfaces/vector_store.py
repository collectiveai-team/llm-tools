from abc import ABC, abstractmethod

import numpy as np

from llm_tools.logger import get_logger
from llm_tools.meta.retrieve_document import Document

logger = get_logger(__name__)


class VectorStore(ABC):
    @abstractmethod
    def load(self, documents: list[Document]) -> list[np.ndarray]:
        pass

    @abstractmethod
    def save(self, documents: list[Document], vectors: np.ndarray) -> None:
        pass

    @abstractmethod
    def search_by_vector(self, query_vector: np.ndarray, k: int = 20) -> list[Document]:
        pass