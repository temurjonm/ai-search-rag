from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    document_id: int
    filename: str
    chunk_count: int
    status: str


class SearchRequest(BaseModel):
    tenant_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class Source(BaseModel):
    document_id: int
    filename: str
    chunk_id: int
    chunk_index: int
    score: float
    excerpt: str


class SearchResponse(BaseModel):
    answer: str
    sources: list[Source]