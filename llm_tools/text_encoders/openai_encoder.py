from typing import Optional

import numpy as np
from openai import OpenAI
from more_itertools import flatten
from tiktoken import encoding_for_model

from llm_tools.logger import get_logger
from llm_tools.meta.retrieve_document import Document
from llm_tools.meta.interfaces.text_encoder import TextEncoder
from llm_tools.meta.interfaces.vector_store import VectorStore

logger = get_logger(__name__)


class OpenAIEncoder(TextEncoder):
    def __init__(
        self,
        batch_size: int = 256,
        max_concurrency: int = 5,
        model_name: str = "text-embedding-3-large",
        dimensions: int = 1024,
        embedding_cache: Optional[VectorStore] = None,
        tokenizer_model: str = "gpt-4o",
    ):
        super().__init__(
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            embedding_cache=embedding_cache,
        )

        self.openai_client = OpenAI()
        self.model_name = model_name
        self.dimensions = dimensions
        self.tokenizer = encoding_for_model(tokenizer_model)

    def _get_n_tokens(self, texts: list[str]) -> int:
        tokens = self.tokenizer.encode_batch(texts)
        return len(list(flatten(tokens)))

    def _encode(self, documents: list[Document]) -> np.ndarray:
        texts = [document.text for document in documents]
        response = self.openai_client.embeddings.create(
            input=texts,
            model=self.model_name,
            dimensions=self.dimensions,
        )

        embeddings = [data_item.embedding for data_item in response.data]
        embeddings = np.array(embeddings)

        return embeddings
