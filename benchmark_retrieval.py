"""
🏥 CLINICAL TRIAL RAG - RETRIEVAL BENCHMARK SCRIPT
--------------------------------------------------
This script compares 3 search methods for Clinical Trial data:
1. Dense Search  : Finds results based on MEANING (Vectors / FastEmbed)
2. Sparse Search : Finds results based on EXACT KEYWORDS (Full-Text Search)
3. Hybrid Search : Combines both Dense and Sparse using Reciprocal Rank Fusion (RRF)

Metrics Calculated:
- ⏱️ Latency (Speed in milliseconds via ThreadPoolExecutor)
- 🎯 Precision@5 (% of top 5 retrieved chunks that are relevant)
- 📥 Recall@5, Recall@10, Recall@20 (% of target keywords captured at cutoffs)
- 🥇 MRR (Mean Reciprocal Rank)
- 📈 NDCG@5 (Normalized Discounted Cumulative Gain at rank 5)
- 📊 Result Overlap (% similarity between Dense and Hybrid results)
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dense import dense_search
from sparse import sparse_search
from hybrid import hybrid_search

# ---------------------------------------------------------
# STEP 1: Define Test Queries with Target Keywords
# ---------------------------------------------------------
TEST_QUERIES = [
    {
        "type": "Exact Keyword Match",
        "query": "Metformin side effects nausea gastrointestinal",
        "condition": "diabetes",
        "target_keywords": ["metformin", "nausea", "adverse", "gastrointestinal", "side"],
    },
    {
        "type": "Rare Term / Drug Name",
        "query": "Pembrolizumab adverse events phase 3",
        "condition": "cancer",
        "target_keywords": ["pembrolizumab", "adverse", "phase"],
    },
    {
        "type": "Conceptual / Meaning-based",
        "query": "blood sugar regulation medications causing stomach discomfort",
        "condition": "diabetes",
        "target_keywords": ["sugar", "glucose", "stomach", "nausea", "discomfort"],
    },
    {
        "type": "Broad Clinical Query",
        "query": "memory loss progression and cognitive decline treatment eligibility",
        "condition": "alzheimers",
        "target_keywords": ["memory", "cognitive", "eligibility", "decline"],
    },
]


def calculate_metrics(results: list[dict], target_keywords: list[str]) -> dict:
    """
    Calculates detailed retrieval metrics across cutoffs (k=5, 10, 20):
    - Precision@5
    - Recall@5, Recall@10, Recall@20
    - MRR (Mean Reciprocal Rank)
    - NDCG@5
    """
    if not results:
        return {
            "p5": 0.0,
            "r5": 0.0,
            "r10": 0.0,
            "r20": 0.0,
            "mrr": 0.0,
            "ndcg5": 0.0,
        }

    def is_relevant(item: dict) -> bool:
        text = (item.get("content", "") + " " + item.get("section", "")).lower()
        return any(kw.lower() in text for kw in target_keywords)

    def calculate_recall_at_n(n: int) -> float:
        subset = results[:n]
        all_text = " ".join([
            (r.get("content", "") + " " + r.get("section", "")).lower()
            for r in subset
        ])
        found = sum(1 for kw in target_keywords if kw.lower() in all_text)
        return (found / len(target_keywords)) * 100.0 if target_keywords else 0.0

    # Precision@5
    top_5 = results[:5]
    relevant_chunks_5 = sum(1 for item in top_5 if is_relevant(item))
    p5 = (relevant_chunks_5 / len(top_5)) * 100.0 if top_5 else 0.0

    # Recall at cutoffs
    r5 = calculate_recall_at_n(5)
    r10 = calculate_recall_at_n(10)
    r20 = calculate_recall_at_n(20)

    # MRR (First relevant rank)
    mrr = 0.0
    for idx, item in enumerate(results, start=1):
        if is_relevant(item):
            mrr = 1.0 / idx
            break

    # NDCG@5
    dcg5 = 0.0
    for idx, item in enumerate(top_5, start=1):
        rel = 1.0 if is_relevant(item) else 0.0
        dcg5 += rel / math.log2(idx + 1)

    total_rel_in_pool = sum(1 for item in results if is_relevant(item))
    ideal_rel_count = min(5, total_rel_in_pool)
    idcg5 = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_rel_count + 1))
    ndcg5 = (dcg5 / idcg5) if idcg5 > 0 else 0.0

    return {
        "p5": p5,
        "r5": r5,
        "r10": r10,
        "r20": r20,
        "mrr": mrr,
        "ndcg5": ndcg5,
    }


def calculate_precision_and_recall(results: list[dict], target_keywords: list[str]) -> tuple[float, float]:
    metrics = calculate_metrics(results, target_keywords)
    return metrics["p5"], metrics["r5"]


def calculate_overlap_percentage(results_a: list[dict], results_b: list[dict]) -> float:
    """
    Calculates what percentage of document titles/sections match between two result lists.
    """
    if not results_a or not results_b:
        return 0.0

    items_a = set(f"{item.get('doc_id')}_{item.get('section')}" for item in results_a[:5])
    items_b = set(f"{item.get('doc_id')}_{item.get('section')}" for item in results_b[:5])

    common_count = len(items_a.intersection(items_b))
    total_unique = len(items_a.union(items_b))

    if total_unique == 0:
        return 0.0

    return (common_count / total_unique) * 100.0


def run_benchmark(candidate_k: int = 30):
    print("=" * 80)
    print(f"🏥 CLINICAL TRIAL RAG RETRIEVAL BENCHMARK (Candidate Pool Size: {candidate_k})")
    print("=" * 80)

    dense_metrics_list, sparse_metrics_list, hybrid_metrics_list = [], [], []
    dense_times, sparse_times, hybrid_times = [], [], []
    dense_vs_hybrid_overlaps = []

    for index, test_case in enumerate(TEST_QUERIES, start=1):
        query_text = test_case["query"]
        query_type = test_case["type"]
        condition = test_case["condition"]
        target_keywords = test_case["target_keywords"]

        print(f"\n--- TEST #{index}: [{query_type}] ---")
        print(f"Query: \"{query_text}\" (Condition: {condition})")

        # Execute Dense, Sparse, and Hybrid searches concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as executor:
            t_start = time.perf_counter()
            f_dense = executor.submit(dense_search, query_text, condition, None, None, 20)
            f_sparse = executor.submit(sparse_search, query_text, condition, None, 20)
            f_hybrid = executor.submit(hybrid_search, query_text, condition, None, None, 20, candidate_k, 60)

            dense_res = f_dense.result()
            t_dense = (time.perf_counter() - t_start) * 1000

            sparse_res = f_sparse.result()
            t_sparse = (time.perf_counter() - t_start) * 1000

            hybrid_res = f_hybrid.result()
            t_hybrid = (time.perf_counter() - t_start) * 1000

        m_dense = calculate_metrics(dense_res, target_keywords)
        m_sparse = calculate_metrics(sparse_res, target_keywords)
        m_hybrid = calculate_metrics(hybrid_res, target_keywords)

        dense_times.append(t_dense)
        sparse_times.append(t_sparse)
        hybrid_times.append(t_hybrid)

        dense_metrics_list.append(m_dense)
        sparse_metrics_list.append(m_sparse)
        hybrid_metrics_list.append(m_hybrid)

        overlap_pct = calculate_overlap_percentage(dense_res, hybrid_res)
        dense_vs_hybrid_overlaps.append(overlap_pct)

        # PRINT METRICS PER QUERY
        print(f"\n⏱️  Latency (Parallel Threads):")
        print(f"   • Dense: {t_dense:.1f} ms | Sparse: {t_sparse:.1f} ms | Hybrid: {t_hybrid:.1f} ms")

        print(f"\n🎯 Precision@5 (Relevance):")
        print(f"   • Dense: {m_dense['p5']:.0f}% | Sparse: {m_sparse['p5']:.0f}% | Hybrid: {m_hybrid['p5']:.0f}%")

        print(f"\n📥 Recall Cutoffs (Concept Coverage):")
        print(f"   • Recall@5  : Dense: {m_dense['r5']:.0f}% | Sparse: {m_sparse['r5']:.0f}% | Hybrid: {m_hybrid['r5']:.0f}%")
        print(f"   • Recall@10 : Dense: {m_dense['r10']:.0f}% | Sparse: {m_sparse['r10']:.0f}% | Hybrid: {m_hybrid['r10']:.0f}%")
        print(f"   • Recall@20 : Dense: {m_dense['r20']:.0f}% | Sparse: {m_sparse['r20']:.0f}% | Hybrid: {m_hybrid['r20']:.0f}%")

        print(f"\n🥇 MRR & NDCG@5:")
        print(f"   • MRR     : Dense: {m_dense['mrr']:.3f} | Sparse: {m_sparse['mrr']:.3f} | Hybrid: {m_hybrid['mrr']:.3f}")
        print(f"   • NDCG@5  : Dense: {m_dense['ndcg5']:.3f} | Sparse: {m_sparse['ndcg5']:.3f} | Hybrid: {m_hybrid['ndcg5']:.3f}")

        print(f"\n📊 Result Overlap (Top 5):")
        print(f"   • Dense vs Hybrid Match: {overlap_pct:.1f}%")

        top_d = f"{dense_res[0]['section']} ({dense_res[0]['doc_id']})" if dense_res else "No result"
        top_s = f"{sparse_res[0]['section']} ({sparse_res[0]['doc_id']})" if sparse_res else "No result"
        top_h = f"{hybrid_res[0]['section']} ({hybrid_res[0]['doc_id']})" if hybrid_res else "No result"
        print(f"\n🥇 #1 Ranked Result by Method:")
        print(f"   • Dense  👉 {top_d}")
        print(f"   • Sparse 👉 {top_s}")
        print(f"   • Hybrid 👉 {top_h}")

    def avg(lst, key):
        return sum(x[key] for x in lst) / len(lst)

    # SUMMARY REPORT
    print("\n" + "=" * 80)
    print(f"📈 OVERALL BENCHMARK SUMMARY REPORT (Candidate Pool Size = {candidate_k})")
    print("=" * 80)
    print(f"1. Precision@5:")
    print(f"   • Dense Search  : {avg(dense_metrics_list, 'p5'):.1f}%")
    print(f"   • Sparse Search : {avg(sparse_metrics_list, 'p5'):.1f}%")
    print(f"   • Hybrid Search : {avg(hybrid_metrics_list, 'p5'):.1f}%")

    print(f"\n2. Recall Cutoffs:")
    print(f"   • Recall@5  : Dense: {avg(dense_metrics_list, 'r5'):.1f}% | Sparse: {avg(sparse_metrics_list, 'r5'):.1f}% | Hybrid: {avg(hybrid_metrics_list, 'r5'):.1f}%")
    print(f"   • Recall@10 : Dense: {avg(dense_metrics_list, 'r10'):.1f}% | Sparse: {avg(sparse_metrics_list, 'r10'):.1f}% | Hybrid: {avg(hybrid_metrics_list, 'r10'):.1f}%")
    print(f"   • Recall@20 : Dense: {avg(dense_metrics_list, 'r20'):.1f}% | Sparse: {avg(sparse_metrics_list, 'r20'):.1f}% | Hybrid: {avg(hybrid_metrics_list, 'r20'):.1f}%")

    print(f"\n3. Ranking Quality (MRR & NDCG@5):")
    print(f"   • MRR    : Dense: {avg(dense_metrics_list, 'mrr'):.3f} | Sparse: {avg(sparse_metrics_list, 'mrr'):.3f} | Hybrid: {avg(hybrid_metrics_list, 'mrr'):.3f}")
    print(f"   • NDCG@5 : Dense: {avg(dense_metrics_list, 'ndcg5'):.3f} | Sparse: {avg(sparse_metrics_list, 'ndcg5'):.3f} | Hybrid: {avg(hybrid_metrics_list, 'ndcg5'):.3f}")

    print(f"\n4. Average Speed (Latency):")
    print(f"   • Dense Search  : {sum(dense_times)/len(dense_times):.1f} ms")
    print(f"   • Sparse Search : {sum(sparse_times)/len(sparse_times):.1f} ms")
    print(f"   • Hybrid Search : {sum(hybrid_times)/len(hybrid_times):.1f} ms")

    avg_overlap = sum(dense_vs_hybrid_overlaps) / len(dense_vs_hybrid_overlaps)
    print(f"\n5. Result Difference (Top 5):")
    print(f"   • On average, Dense and Hybrid share {avg_overlap:.1f}% of top 5 results.")
    print(f"   • That means Hybrid search brings in {100 - avg_overlap:.1f}% NEW relevant results!")
    print("=" * 80)


def run_candidate_experiments(candidate_sizes=[20, 30, 50]):
    print("\n" + "#" * 80)
    print("🔬 EXPERIMENTING WITH DENSE & SPARSE RETRIEVAL CANDIDATE POOL SIZES")
    print("#" * 80)
    
    summary_rows = []
    for candidate_k in candidate_sizes:
        print(f"\n>>> TESTING CANDIDATE POOL SIZE: {candidate_k}")
        dense_m, sparse_m, hybrid_m = [], [], []
        h_times = []

        for test_case in TEST_QUERIES:
            query_text = test_case["query"]
            condition = test_case["condition"]
            target_keywords = test_case["target_keywords"]

            with ThreadPoolExecutor(max_workers=3) as executor:
                t_start = time.perf_counter()
                f_dense = executor.submit(dense_search, query_text, condition, None, None, 20)
                f_sparse = executor.submit(sparse_search, query_text, condition, None, 20)
                f_hybrid = executor.submit(hybrid_search, query_text, condition, None, None, 20, candidate_k, 60)

                d_res = f_dense.result()
                s_res = f_sparse.result()
                h_res = f_hybrid.result()
                t_h = (time.perf_counter() - t_start) * 1000

            m_d = calculate_metrics(d_res, target_keywords)
            m_s = calculate_metrics(s_res, target_keywords)
            m_h = calculate_metrics(h_res, target_keywords)

            dense_m.append(m_d)
            sparse_m.append(m_s)
            hybrid_m.append(m_h)
            h_times.append(t_h)

        def avg(lst, k):
            return sum(x[k] for x in lst) / len(lst)

        summary_rows.append({
            "candidate_k": candidate_k,
            "p5": avg(hybrid_m, "p5"),
            "r5": avg(hybrid_m, "r5"),
            "r10": avg(hybrid_m, "r10"),
            "r20": avg(hybrid_m, "r20"),
            "mrr": avg(hybrid_m, "mrr"),
            "ndcg5": avg(hybrid_m, "ndcg5"),
            "latency": sum(h_times) / len(h_times),
        })

    print("\n" + "=" * 85)
    print("📊 CANDIDATE RETRIEVAL POOL SIZE EXPERIMENT RESULTS (HYBRID SEARCH)")
    print("=" * 85)
    print(f"{'Candidate Size':<16} | {'P@5 (%)':<9} | {'R@5 (%)':<9} | {'R@10 (%)':<10} | {'R@20 (%)':<10} | {'MRR':<7} | {'NDCG@5':<8} | {'Latency (ms)'}")
    print("-" * 85)
    for r in summary_rows:
        print(f"{r['candidate_k']:<16} | {r['p5']:<9.1f} | {r['r5']:<9.1f} | {r['r10']:<10.1f} | {r['r20']:<10.1f} | {r['mrr']:<7.3f} | {r['ndcg5']:<8.3f} | {r['latency']:.1f} ms")
    print("=" * 85)


if __name__ == "__main__":
    run_benchmark(candidate_k=30)
    run_candidate_experiments(candidate_sizes=[20, 30, 50])


