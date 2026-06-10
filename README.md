# 🏥 Clinical Trial Navigator

A hybrid RAG (Retrieval-Augmented Generation) system for searching, analyzing, and comparing clinical trial data. Built with semantic vector search, full-text keyword search, contradiction detection, and Gemini 2.0 Flash for grounded answer generation.

---

## 📌 Problem Statement

Researchers, clinicians, and patients struggle to extract meaningful insights from thousands of clinical trial documents. Manually reading trial reports to find drug side effects, eligibility criteria, or outcome comparisons is slow, error-prone, and inaccessible to non-experts.

This project solves that by building an intelligent search and QA system on top of real clinical trial data — combining the precision of keyword search with the intelligence of semantic search and LLM-powered answer generation.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
│                     Streamlit Web App                        │
│         Research Chat  │  Adverse Events Analysis           │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │       hybrid.py            │
                    │   Hybrid Search (RRF)      │
                    └──────┬──────────┬──────────┘
                           │          │
             ┌─────────────▼──┐  ┌───▼──────────────┐
             │   dense.py     │  │    sparse.py      │
             │ Semantic Search│  │  FTS Keyword      │
             │ HNSW + vectors │  │  Search tsquery   │
             └─────────────┬──┘  └───┬───────────────┘
                           │          │
                    ┌──────▼──────────▼──────┐
                    │       Supabase          │
                    │   clinical_chunks       │
                    │  embedding vector(384)  │
                    │  GIN FTS index          │
                    │  HNSW index             │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼─────────────┐
                    │        llm.py           │
                    │   Gemini 2.0 Flash      │
                    │   Answer Generation     │
                    └──────────┬─────────────┘
                               │
          ┌────────────────────▼──────────────────┐
          │        contradiction_detector.py        │
          │   Cross-trial Contradiction Analysis    │
          └───────────────────────────────────────┘

─────────────────────────────────────────────────
                  INGESTION PIPELINE
─────────────────────────────────────────────────

  ClinicalTrials.gov V2 API
          │
          ▼
      fetcher.py          pulls trials by condition
          │
          ▼
      parser.py           splits into 5 sections per trial
          │               Summary, Description, Eligibility,
          │               Outcomes, AdverseEvents
          ▼
      embedder.py         embeds with BAAI/bge-small-en-v1.5
          │               stores in Supabase with embeddings
          ▼
      Supabase DB         clinical_chunks table populated
```

---

## ⚙️ How It Works

### 1. Data Ingestion
- Fetches real clinical trial data from the **ClinicalTrials.gov V2 API** for 5 conditions: diabetes, cancer, hypertension, alzheimers, cardiovascular disease
- Each trial is split into exactly **5 sections**: Summary, Description, Eligibility, Outcomes, AdverseEvents
- Each section is embedded using **BAAI/bge-small-en-v1.5** (384 dimensions) via FastEmbed
- Stored in a single **Supabase PostgreSQL** table with both vector embeddings and full-text search indexes

### 2. Hybrid Search
- **Dense search** — converts the user query into a 384d vector and finds semantically similar chunks using the HNSW cosine index
- **Sparse search** — converts the query into a PostgreSQL tsquery and finds keyword-matching chunks using the GIN full-text index
- **RRF fusion** — Reciprocal Rank Fusion combines both result lists. Chunks appearing in both lists score higher

### 3. Answer Generation
- Top retrieved chunks are passed as context to **Gemini 2.0 Flash**
- The model generates a grounded answer citing Trial IDs
- It never makes up information — if context is insufficient it says so

### 4. Contradiction Detection
- Retrieved chunks are grouped by trial
- Gemini analyzes excerpts from multiple trials to find conflicting claims
- Contradictions are classified as **Minor**, **Significant**, or **Critical**
- Shown to the user with side-by-side trial comparisons

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| Web Framework | Streamlit |
| Database | Supabase (PostgreSQL + pgvector) |
| Vector Search | pgvector HNSW index |
| Keyword Search | PostgreSQL Full Text Search (GIN index) |
| Embedding Model | BAAI/bge-small-en-v1.5 via FastEmbed |
| LLM | Gemini 2.0 Flash (Google AI) |
| HTTP Client | httpx |
| Environment | python-dotenv |
| Data Source | ClinicalTrials.gov V2 API |

---

## 📂 Project Structure

```
Clinic_Rag/
├── .env                        # API keys (never commit this)
├── config.py                   # Loads environment variables
├── database.py                 # Supabase client singleton
├── fetcher.py                  # ClinicalTrials.gov API fetcher
├── parser.py                   # Splits trials into sections
├── embedder.py                 # Embeds and stores in Supabase
├── ingest_trials.py            # Main ingestion runner script
├── dense.py                    # Semantic vector search
├── sparse.py                   # FTS keyword search
├── hybrid.py                   # RRF fusion of both searches
├── llm.py                      # Gemini answer generation
├── contradiction_detector.py   # Cross-trial contradiction analysis
├── streamlit_app.py            # Streamlit web interface
└── test_connection.py          # Database connection test
```

---

## 🗄️ Database Schema

```sql
table: clinical_chunks
├── id            bigint (primary key, auto)
├── doc_id        text    (NCT trial ID)
├── doc_type      text    (always "clinical_trial")
├── title         text    (trial title)
├── source_id     text    (same as doc_id)
├── condition     text    (e.g. "diabetes")
├── drug_name     text    (e.g. "Metformin")
├── phase         text    (e.g. "Phase 3")
├── status        text    (e.g. "Completed")
├── content       text    (section text with metadata header)
├── section       text    (Summary / Description / Eligibility / Outcomes / AdverseEvents)
├── chunk_index   integer (0 to 4)
├── embedding     vector(384)  ← used by HNSW dense search
└── created_at    timestamptz

indexes:
├── idx_embedding_hnsw   HNSW cosine index on embedding   ← dense search
├── idx_content_fts      GIN tsvector index on content    ← sparse search
├── idx_condition        btree on condition
├── idx_drug_name        btree on drug_name
└── idx_phase            btree on phase

functions:
├── search_clinical_chunks()      ← dense search RPC
└── search_clinical_chunks_fts()  ← sparse search RPC
```

---

## 📊 Data Sources

| Source | Type | Access |
|---|---|---|
| ClinicalTrials.gov V2 API | REST API | Public, no key needed |
| Conditions covered | diabetes, cancer, hypertension, alzheimers, cardiovascular disease | — |
| Trials per condition | up to 10 completed trials with results | — |
| Sections per trial | 5 (Summary, Description, Eligibility, Outcomes, AdverseEvents) | — |

---

## 🚀 How To Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/clinic-rag.git
cd clinic-rag
```

**2. Install dependencies**
```bash
pip install streamlit supabase httpx fastembed rank-bm25 python-dotenv google-generativeai
```

**3. Set up your `.env` file**
```
GOOGLE_API_KEY=your_gemini_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

**4. Set up Supabase database**

Go to your Supabase SQL editor and run the full SQL setup from `supabase_setup.sql` which creates the table, indexes, and both RPC functions.

**5. Run the ingestion pipeline**
```bash
python ingest_trials.py
```
Wait for it to finish. Verify rows appear in your Supabase table with non-null embeddings.

**6. Test retrieval**
```bash
python dense.py
python sparse.py
python hybrid.py
```

**7. Launch the app**
```bash
streamlit run streamlit_app.py
```

---

## 💡 Features

| Feature | Description |
|---|---|
| 🔍 Hybrid Search | Combines semantic + keyword search using RRF for better results |
| 🤖 AI Answer Generation | Gemini 2.0 Flash generates grounded answers from retrieved context |
| ⚠️ Contradiction Detection | Flags conflicting claims across trials with severity ratings |
| 💊 Adverse Events Analysis | Dedicated tab for drug side effect analysis |
| 🔎 Search Filters | Filter by condition, drug name, phase, and number of results |
| 📚 Source Citations | Every answer shows the Trial IDs used as sources |
| 🗂️ View Sources | Expandable section showing raw chunks used for each answer |

---

## 🔮 Future Improvements

- **RAGAS Evaluation** — measure faithfulness, context recall, and answer relevancy scores
- **Trial Comparison** — side by side comparison of two drugs or conditions
- **Confidence Score** — show similarity scores so users know retrieval quality
- **Export to PDF** — download answers as formatted PDF reports
- **More Conditions** — extend ingestion to cover more diseases
- **Reranking** — add a cross-encoder reranker after hybrid search for better precision

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)

---

## 📄 License

MIT License — free to use and modify.
