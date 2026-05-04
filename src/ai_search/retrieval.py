from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from ai_search.models import Chunk, Document
from ai_search.schemas import Source


@dataclass(frozen=True)
class Candidate:
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    content: str
    rank: int
    score: float


def reciprocal_rank_fusion(
    ranked_lists: list[list[Candidate]],
    limit: int,
    k: int = 60,
) -> list[Candidate]:
    by_chunk: dict[int, Candidate] = {}
    scores: dict[int, float] = {}

    for candidates in ranked_lists:
        for rank, candidate in enumerate(candidates, start=1):
            by_chunk[candidate.chunk_id] = candidate
            scores[candidate.chunk_id] = scores.get(candidate.chunk_id, 0.0) + (1.0 / (k + rank))

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        Candidate(
            chunk_id=by_chunk[chunk_id].chunk_id,
            document_id=by_chunk[chunk_id].document_id,
            filename=by_chunk[chunk_id].filename,
            chunk_index=by_chunk[chunk_id].chunk_index,
            content=by_chunk[chunk_id].content,
            rank=index + 1,
            score=score,
        )
        for index, (chunk_id, score) in enumerate(fused[:limit])
    ]


def vector_search(
    db: Session,
    tenant_id: str,
    query_embedding: list[float],
    limit: int,
) -> list[Candidate]:
    distance = Chunk.embedding.cosine_distance(query_embedding)
    rows = (
        db.query(
            Chunk.id,
            Chunk.document_id,
            Chunk.chunk_index,
            Chunk.content,
            distance.label("distance"),
        )
        .filter(Chunk.tenant_id == tenant_id)
        .order_by(distance)
        .limit(limit)
        .all()
    )

    document_names = _document_names(db, [row.document_id for row in rows])
    return [
        Candidate(
            chunk_id=row.id,
            document_id=row.document_id,
            filename=document_names.get(row.document_id, "unknown"),
            chunk_index=row.chunk_index,
            content=row.content,
            rank=index + 1,
            score=1.0 - float(row.distance),
        )
        for index, row in enumerate(rows)
    ]


def keyword_search(db: Session, tenant_id: str, query: str, limit: int) -> list[Candidate]:
    sql = text(
        """
        SELECT
            c.id AS chunk_id,
            c.document_id AS document_id,
            d.filename AS filename,
            c.chunk_index AS chunk_index,
            c.content AS content,
            ts_rank_cd(
                to_tsvector('english', c.content),
                websearch_to_tsquery('english', :query)
            ) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.tenant_id = :tenant_id
          AND websearch_to_tsquery('english', :query) @@ to_tsvector('english', c.content)
        ORDER BY score DESC
        LIMIT :limit
        """
    )
    rows = db.execute(sql, {"tenant_id": tenant_id, "query": query, "limit": limit}).mappings().all()

    return [
        Candidate(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            filename=row["filename"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            rank=index + 1,
            score=float(row["score"]),
        )
        for index, row in enumerate(rows)
    ]


def hybrid_search(
    db: Session,
    tenant_id: str,
    question: str,
    question_embedding: list[float],
    top_k: int,
) -> list[Source]:
    candidate_limit = max(top_k * 4, 20)
    vector_candidates = vector_search(db, tenant_id, question_embedding, candidate_limit)
    keyword_candidates = keyword_search(db, tenant_id, question, candidate_limit)
    fused = reciprocal_rank_fusion([vector_candidates, keyword_candidates], limit=top_k)

    return [
        Source(
            document_id=candidate.document_id,
            filename=candidate.filename,
            chunk_id=candidate.chunk_id,
            chunk_index=candidate.chunk_index,
            score=round(candidate.score, 6),
            excerpt=candidate.content[:900],
        )
        for candidate in fused
    ]


def _document_names(db: Session, document_ids: list[int]) -> dict[int, str]:
    if not document_ids:
        return {}

    rows = db.query(Document.id, Document.filename).filter(Document.id.in_(document_ids)).all()
    return {row.id: row.filename for row in rows}