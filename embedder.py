from fastembed import TextEmbedding
from database import get_supabase

# Load the embedding model once at module level, never inside a function
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
db = get_supabase()


def embed_texts(texts: list[str]) -> list[list[float]]:
    results = embedding_model.embed(texts)
    return [embedding.tolist() for embedding in results]


def store_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    rows = []
    for chunk, embedding in zip(chunks, embeddings):
        rows.append({
            "doc_id":      chunk["doc_id"],
            "doc_type":    chunk["doc_type"],
            "title":       chunk["title"],
            "source_id":   chunk["source_id"],
            "condition":   chunk["condition"],
            "drug_name":   chunk["drug_name"],
            "phase":       chunk["phase"],
            "status":      chunk["status"],
            "content":     chunk["content"],
            "section":     chunk["section"],
            "chunk_index": chunk["chunk_index"],
            "embedding":   embedding,
        })

    for i in range(0, len(rows), 50):
        batch = rows[i : i + 50]
        db.table("clinical_chunks").insert(batch).execute()
        print(f"  ✅ Inserted batch of {len(batch)} rows")


def ingest_trial(trial_info: dict, condition: str) -> None:
    from parser import trial_to_sections

    try:
        # Get all 5 section rows from the trial (including AdverseEvents)
        all_chunks = trial_to_sections(trial_info, condition)
        if not all_chunks:
            return

        # Embed and store all chunks
        texts = [chunk["content"] for chunk in all_chunks]
        embeddings = embed_texts(texts)
        store_chunks(all_chunks, embeddings)

        nct_id = all_chunks[0]["doc_id"]
        title  = all_chunks[0]["title"]
        print(f"  Ingested {nct_id}: {title[:60]}")

    except Exception as e:
        protocol = trial_info.get("ProtocolSection", {})
        id_module = protocol.get("IdentificationModule", {})
        nct_id = id_module.get("NCTId", "UNKNOWN")
        print(f"  ⚠ Skipping {nct_id}: {e}")

