from abc import ABC

import numpy as np

from llm_tools.logger import get_logger
from llm_tools.meta.retrieve_document import Document
from llm_tools.meta.interfaces.text_encoder import TextEncoder

logger = get_logger(__name__)


class Retrieval(ABC):
    def __init__(self, encoder: TextEncoder, collection_name: str):
        self.encoder = encoder
        self.collection_name = collection_name

    def add_documents(self, documents: list[Document]) -> None:
        self.encoder.batch_encode(documents)

    def retrieve(
        self,
        query: str,
        k: int = 20,
        filter: dict = None,
    ) -> list[Document]:
        encoded_query = self.encoder.encode([Document(text=query)], force_no_save=True)[
            0
        ]
        return self.encoder.embedding_cache.search_by_vector(encoded_query, k)

    def retrieve_by_vector(
        self,
        query_vector: np.ndarray,
        k: int = 20,
        filter: dict = None,
    ) -> list[Document]:

        if filter:
            raise NotImplementedError("Filtering not supported")
        return self.encoder.embedding_cache.search_by_vector(query_vector, k)
