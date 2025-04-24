from typing import Any, Dict, List, Literal
from pydantic import Field

from app.schemas import BaseModel


class Chunk(BaseModel):
    object: Literal["chunk"] = "chunk"
    id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content: str


class Chunks(BaseModel):
    object: Literal["list"] = "list"
    data: List[Chunk]
