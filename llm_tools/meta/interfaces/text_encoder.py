import asyncio
from abc import ABC, abstractmethod
from typing import Iterator, Optional

import numpy as np
from tqdm import tqdm
from more_itertools import chunked, flatten

from llm_tools.logger import get_logger
from llm_tools.meta.retrieve_document import Document
from llm_tools.meta.interfaces.vector_store import VectorStore

logger = get_logger(__name__)


class TextEncoder(ABC):
    def __init__(
        self,
        batch_size: int = 1024,
        max_concurrency: int = 10,
        embedding_cache: Optional[VectorStore] = None,
    ):
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.embedding_cache = embedding_cache

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def _get_n_tokens(self, texts: list[str]) -> int:
        pass

    @abstractmethod
    def _encode(self, documents: list[Document]) -> np.ndarray:
        pass

    def encode(
        self, documents: list[Document], force_no_save: bool = False
    ) -> np.ndarray:

        if self.embedding_cache is None or force_no_save:
            return self._encode(documents=documents)

        loaded_vectors = self.embedding_cache.load(documents=documents)
        uncached_indexes = {
            idx for idx, vector in enumerate(loaded_vectors) if vector is None
        }

        if not uncached_indexes:
            return np.array(loaded_vectors)

        uncached_documents = [
            text for idx, text in enumerate(documents) if idx in uncached_indexes
        ]

        uncached_vectors = self._encode(documents=uncached_documents)
        self.embedding_cache.save(
            documents=uncached_documents,
            vectors=uncached_vectors,
        )

        for idx, uncached_vector in zip(uncached_indexes, uncached_vectors):
            loaded_vectors[idx] = uncached_vector

        assert len(documents) == len(loaded_vectors)
        return np.array(loaded_vectors)

    def batch_encode(self, documents: list[Document]) -> Iterator[np.ndarray]:
        texts = [doc.text for doc in documents]
        n_tokens = self._get_n_tokens(texts=texts)
        if len(documents) <= self.batch_size:
            return self.encode(documents=documents)

        documents_chunks = chunked(documents, self.batch_size)
        chunk_vectors = map(
            self.encode,
            tqdm(
                documents_chunks,
                total=(len(documents) // self.batch_size),
                desc=f"encoding {n_tokens} tokens",
                ascii=" ##",
                colour="#808080",
            ),
        )

        text_vetors = flatten(chunk_vectors)
        return text_vetors

    async def async_encode(
        self,
        documents: list[Document],
        pbar: tqdm,
    ) -> np.ndarray:
        async with self.semaphore:
            vectors = await asyncio.to_thread(self.encode, documents)
            pbar.update(1)

            return vectors

    async def async_batch_encode(self, documents: list[Document]) -> np.ndarray:
        texts = [doc.text for doc in documents]
        n_tokens = self._get_n_tokens(texts=texts)
        if len(documents) <= self.batch_size:
            return self.encode(documents=documents)

        documents_chunks = chunked(documents, self.batch_size)
        with tqdm(
            documents_chunks,
            total=(len(documents) // self.batch_size),
            desc=f"encoding {n_tokens} tokens",
            ascii=" ##",
            colour="#808080",
        ) as pbar:
            async_tasks = [
                self.async_encode(documents, pbar=pbar)
                for documents in documents_chunks
            ]

            chunk_vectors = await asyncio.gather(*async_tasks)

        text_vetors = np.array(list(flatten(chunk_vectors)))
        return text_vetors
