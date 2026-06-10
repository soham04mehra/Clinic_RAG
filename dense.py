from fastembed import TextEmbedding
from database import get_supabase

embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")


def dense_search(
    query: str,
    condition: str = None,
    drug_name: str = None,
    phase: str = None,
    top_k: int = 10,
) -> list[dict]:
    query_vector = list(embedding_model.embed([query]))[0].tolist()
    db = get_supabase()

    response = db.rpc(
        "search_clinical_chunks",
        {
            "query_embedding": query_vector,
            "condition_filter": condition,
            "drug_filter": drug_name,
            "phase_filter": phase,
            "match_count": top_k,
        },
    ).execute()

    results = response.data or []

    for i, r in enumerate(results):
        r["rank"] = i + 1
        r["retriever"] = "dense"

    return results


if __name__ == "__main__":
    results = dense_search(
        query="what are the side effects of metformin", condition="diabetes"
    )
    for r in results:
        print(
            f"Rank {r['rank']} | {r['similarity']:.3f} | {r['section']} | {r['title'][:50]}"
        )
