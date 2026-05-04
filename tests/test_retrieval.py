from ai_search.retrieval import Candidate, reciprocal_rank_fusion


def candidate(chunk_id: int, rank: int) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        document_id=chunk_id,
        filename=f"doc-{chunk_id}.txt",
        chunk_index=0,
        content=f"content {chunk_id}",
        rank=rank,
        score=1.0 / rank,
    )


def test_rrf_boosts_candidates_seen_in_multiple_lists() -> None:
    dense = [candidate(1, 1), candidate(2, 2), candidate(3, 3)]
    keyword = [candidate(3, 1), candidate(1, 2), candidate(4, 3)]

    fused = reciprocal_rank_fusion([dense, keyword], limit=3)

    assert [item.chunk_id for item in fused] == [1, 3, 2]


def test_rrf_respects_limit() -> None:
    dense = [candidate(1, 1), candidate(2, 2), candidate(3, 3)]

    fused = reciprocal_rank_fusion([dense], limit=2)

    assert len(fused) == 2