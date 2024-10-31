import os

import numpy as np
import weaviate.classes as wvc
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from weaviate.config import AdditionalConfig, ConnectionConfig
from weaviate.classes.config import DataType, Property, Tokenization

from llm_tools.logger import get_logger
from llm_tools.meta.retrieve_document import Document
from llm_tools.meta.interfaces.vector_store import VectorStore

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))


logger = get_logger(__name__)


class WeaviateStore(VectorStore):
    def __init__(
        self,
        collection_name: str = "cache",
        weaviate_host: str = WEAVIATE_HOST,
        weaviate_port: int = WEAVIATE_PORT,
        weaviate_grpc_port: int = WEAVIATE_GRPC_PORT,
        max_connections: int = 128,
    ):
        self.weaviate_client = WeaviateClient(
            connection_params=ConnectionParams.from_params(
                http_host=weaviate_host,
                http_port=weaviate_port,
                http_secure=False,
                grpc_host=weaviate_host,
                grpc_port=weaviate_grpc_port,
                grpc_secure=False,
            ),
            additional_config=AdditionalConfig(
                connection=ConnectionConfig(
                    session_pool_connections=max_connections,
                    session_pool_maxsize=max_connections,
                )
            ),
            skip_init_checks=True,
        )

        self.weaviate_client.connect()
        self.collection_name = collection_name
        self._crate_collection()

    def __del__(self) -> None:
        self.weaviate_client.close()

    def _list_collections(self) -> set[str]:
        collections = self.weaviate_client.collections.list_all()
        return set(collections.keys())

    def _crate_collection(self) -> None:
        if self.collection_name.lower() in [
            collection_name.lower() for collection_name in self._list_collections()
        ]:
            return

        logger.info(f"creating weaviate_collection => {self.collection_name}")
        self.weaviate_client.collections.create(
            name=self.collection_name,
            properties=[
                Property(
                    name="text",
                    data_type=DataType.TEXT,
                    skip_vectorization=True,
                    tokenization=Tokenization.WORD,
                )
            ],
        )

    def _clear_cache(self) -> None:
        logger.warning(f"deleting weaviate collection => {self.collection_name}")
        self.weaviate_client.collections.delete(self.collection_name)

    def save(self, documents: list[Document], vectors: np.ndarray) -> None:
        documents_len, vector_len = len(documents), len(vectors)
        assert documents_len == vector_len, (
            "the length of texts and vectors doesn't match: "
            f"{documents_len} != {vector_len}"
        )

        data = [
            wvc.data.DataObject(
                uuid=document.id,
                vector=vector.tolist(),
                properties={"text": document.text} | document.metadata,
            )
            for vector, document in zip(vectors, documents)
        ]

        collection = self.weaviate_client.collections.get(self.collection_name)
        insert_result = collection.data.insert_many(data)
        assert len(insert_result.uuids) == vector_len

    def load(self, documents: list[Document]) -> list[np.ndarray]:
        collection = self.weaviate_client.collections.get(self.collection_name)
        # uuids = (generate_uuid5(text) for text in texts)
        uuids = [document.id for document in documents]
        wv_objects = (
            collection.query.fetch_object_by_id(uuid=uuid, include_vector=True)
            for uuid in uuids
        )

        loaded_vectors = [
            wv_object.vector["default"] if wv_object is not None else None
            for wv_object in wv_objects
        ]

        assert len(documents) == len(loaded_vectors)
        return loaded_vectors

    # TODO implement a vector search
    def search_by_vector(self, vector: np.ndarray, n_results: int = 5) -> list[Document]:
        collection = self.weaviate_client.collections.get(self.collection_name)
        search_result = collection.query.near_vector(
            near_vector=vector.tolist(), limit=n_results
        )

        return [
            Document(
                id=str(obj.uuid),
                text=obj.properties["text"],
                metadata={k: v for k, v in obj.properties.items() if k != "text"},
            )
            for obj in search_result.objects
        ]

    def clean_collection(self):
        self.weaviate_client.collections.delete(self.collection_name)
        self._crate_collection()