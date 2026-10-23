from dense import dense_search
from sparse import sparse_search
from concurrent.futures import ThreadPoolExecutor


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    scores = {}

    def get_chunk_key(result: dict) -> str:
        if result.get("id"):
            return str(result["id"])
        return f"{result.get('doc_id')}_{result.get('section')}_{result.get('content', '')[:30]}"

    for result in dense_results:
        key = get_chunk_key(result)
        rrf_score = 1.0 / (k + result["rank"])
        if key in scores:
            scores[key]["score"] += rrf_score
        else:
            scores[key] = {"score": rrf_score, "data": result}

    for result in sparse_results:
        key = get_chunk_key(result)
        rrf_score = 1.0 / (k + result["rank"])
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


# Common clinical synonyms to bridge the vocabulary gap between patient queries and trial texts
CLINICAL_SYNONYMS = {
    "blood sugar": "glucose glycemic",
    "stomach discomfort": "gastrointestinal nausea vomiting diarrhea",
    "memory loss": "cognitive impairment dementia alzheimers",
    "side effects": "adverse events toxicity safety",
}

def expand_query(query: str) -> str:
    """Expand common query phrases with clinical terminology for broader retrieval coverage."""
    expanded = query.lower()
    for phrase, synonyms in CLINICAL_SYNONYMS.items():
        if phrase in expanded:
            expanded += f" {synonyms}"
    return expanded


def hybrid_search(
    query: str,
    condition: str = None,
    drug_name: str = None,
    phase: str = None,
    top_k: int = 5,
    candidate_k: int = 30,
    rrf_k: int = 60,
) -> list[dict]:
    search_query = expand_query(query)
    fetch_k = max(candidate_k, top_k)
    # Run Dense and Sparse search concurrently in parallel threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_dense = executor.submit(
            dense_search, search_query, condition, drug_name, phase, fetch_k
        )
        future_sparse = executor.submit(
            sparse_search, search_query, condition, drug_name, fetch_k
        )

        dense_results = future_dense.result()
        sparse_results = future_sparse.result()

    return reciprocal_rank_fusion(dense_results, sparse_results, k=rrf_k, top_k=top_k)


if __name__ == "__main__":
    results = hybrid_search(
        query="what are the side effects of metformin",
        condition="diabetes",
        top_k=5,
    )
    for r in results:
        print(f"Rank {r['rank']} | RRF {r['rrf_score']:.4f} | {r['section']} | {r['title'][:50]}")

