import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import chromadb
import numpy as np

from llm_tools.logger import get_logger
from llm_tools.meta.retrieve_document import Document
from llm_tools.meta.interfaces.vector_store import VectorStore

logger = get_logger(__name__)


class ChromaStore(VectorStore):
    def __init__(
        self,
        collection_name: str = "cache",
        persist_directory: str = "./chroma_cache",
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)

        self.collection_name = collection_name
        self.collection = self._create_collection()

    def _create_collection(self):
        collections = self.client.list_collections()
        existing = [c for c in collections if c.name == self.collection_name]

        if existing:
            return existing[0]

        logger.info(f"Creating new collection => {self.collection_name}")
        return self.client.create_collection(name=self.collection_name)

    def _clear_cache(self) -> None:
        logger.warning(f"deleting chroma collection => {self.collection_name}")
        self.client.delete_collection(self.collection_name)

    def save(self, documents: list[Document], vectors: np.ndarray) -> None:
        documents_len, vector_len = len(documents), len(vectors)
        assert documents_len == vector_len, (
            "the length of texts and vectors doesn't match: "
            f"{documents_len} != {vector_len}"
        )
        ids = [document.id for document in documents]
        texts = [document.text for document in documents]
        metadatas = [
            document.metadata if document.metadata else {"id": str(document.id)}
            for document in documents
        ]

        self.collection.add(
            embeddings=vectors.tolist(), documents=texts, ids=ids, metadatas=metadatas
        )

    def load(self, documents: list[Document]) -> list[np.ndarray]:
        ids = [document.id for document in documents]
        results = self.collection.get(ids=ids, include=["embeddings"])

        loaded_vectors = []
        for idx in range(len(documents)):
            try:
                vector = results["embeddings"][idx]
                loaded_vectors.append(vector if vector is not None else None)
            except (IndexError, KeyError):
                loaded_vectors.append(None)

        assert len(documents) == len(loaded_vectors)
        return loaded_vectors

    def search_by_vector(self, vector: np.ndarray, n_results: int = 5):
        results = self.collection.query(
            query_embeddings=[vector.tolist()], n_results=n_results
        )
        return results
