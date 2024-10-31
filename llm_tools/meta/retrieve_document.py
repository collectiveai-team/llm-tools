import uuid as uuid_lib
from typing import Any, Dict, Optional

from pydantic import Field, BaseModel


def generate_uuid5(identifier: Any, namespace: Any = "") -> str:
    """
    Generate an UUIDv5, may be used to consistently generate the same UUID for a specific
    identifier and namespace.

    Parameters
    ----------
    identifier : Any
        The identifier/object that should be used as basis for the UUID.
    namespace : Any, optional
        Allows to namespace the identifier, by default ""

    Returns
    -------
    str
        The UUID as a string.
    """
    return str(uuid_lib.uuid5(uuid_lib.NAMESPACE_DNS, str(namespace) + str(identifier)))


class Document(BaseModel):
    text: str
    id: Optional[str] = None
    metadata: Optional[Dict] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        if self.id is None:
            self.id = generate_uuid5(self.text)
