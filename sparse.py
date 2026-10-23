import re
from database import get_supabase

# Common English stop words to filter from sparse query so FTS matches content terms effectively
STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "com", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "where", "who", "will", "with", "causes", "causing"
}


def sparse_search(
    query: str,
    condition: str = None,
    drug_name: str = None,
    top_k: int = 30,
) -> list[dict]:
    # Extract words (alpha-numeric)
    raw_words = re.findall(r"\w+", query.lower())
    filtered_words = [w for w in raw_words if w not in STOP_WORDS]
    words_to_use = filtered_words if filtered_words else raw_words

    # Combine words with OR '|' so any matching key term is found
    ts_query = " | ".join(words_to_use) if words_to_use else query.lower()

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
        query="metformin side effects nausea", condition="diabetes", top_k=10
    )
    for r in results:
        print(f"Rank {r['rank']} | Score {r['fts_rank']:.3f} | {r['section']} | {r['title'][:50]}")

