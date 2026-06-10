from dense import dense_search
from sparse import sparse_search


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    scores = {}

    for result in dense_results:
        key = str(result["doc_id"]) + "_" + result["content"][:50]
        rrf_score = 1 / (k + result["rank"])
        if key in scores:
            scores[key]["score"] += rrf_score
        else:
            scores[key] = {"score": rrf_score, "data": result}

    for result in sparse_results:
        key = str(result["doc_id"]) + "_" + result["content"][:50]
        rrf_score = 1 / (k + result["rank"])
        if key in scores:
            scores[key]["score"] += rrf_score
        else:
            scores[key] = {"score": rrf_score, "data": result}

    sorted_entries = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    top_entries = sorted_entries[:top_k]

    results = []
    for idx, entry in enumerate(top_entries):
        data = entry["data"]
        results.append({
            "id":        data.get("id"),
            "doc_id":    data.get("doc_id"),
            "title":     data.get("title"),
            "content":   data.get("content"),
            "section":   data.get("section"),
            "condition": data.get("condition"),
            "drug_name": data.get("drug_name"),
            "phase":     data.get("phase"),
            "rrf_score": entry["score"],
            "rank":      idx + 1,
            "retriever": "hybrid",
        })

    return results


def hybrid_search(
    query: str,
    condition: str = None,
    drug_name: str = None,
    phase: str = None,
    top_k: int = 5,
) -> list[dict]:
    dense_results = dense_search(
        query=query,
        condition=condition,
        drug_name=drug_name,
        phase=phase,
        top_k=10,
    )

    sparse_results = sparse_search(
        query=query,
        condition=condition,
        drug_name=drug_name,
        top_k=10,
    )

    return reciprocal_rank_fusion(dense_results, sparse_results, top_k=top_k)


if __name__ == "__main__":
    results = hybrid_search(
        query="what are the side effects of metformin",
        condition="diabetes",
        top_k=5,
    )
    for r in results:
        print(f"Rank {r['rank']} | RRF {r['rrf_score']:.4f} | {r['section']} | {r['title'][:50]}")
