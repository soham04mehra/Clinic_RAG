from database import get_supabase


def sparse_search(
    query: str,
    condition: str = None,
    drug_name: str = None,
    top_k: int = 10,
) -> list[dict]:
    ts_query = " & ".join(query.lower().split())

    db = get_supabase()
    response = db.rpc("search_clinical_chunks_fts", {
        "query_text": ts_query,
        "condition_filter": condition,
        "drug_filter": drug_name,
        "match_count": top_k,
    }).execute()

    results = response.data or []

    for i, r in enumerate(results):
        r["rank"] = i + 1
        r["retriever"] = "sparse"

    return results


if __name__ == "__main__":
    results = sparse_search(
        query="metformin side effects nausea", condition="diabetes"
    )
    for r in results:
        print(f"Rank {r['rank']} | Score {r['fts_rank']:.3f} | {r['section']} | {r['title'][:50]}")
