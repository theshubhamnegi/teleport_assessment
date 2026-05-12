# Teleport Assessment – Context-Aware Retrieval Engine

I built a small, local RAG pipeline and put two retrieval strategies head-to-head to see which one actually works better in practice.

## The face-off

- Strategy A (Raw Vector Search): Straight-up embedding similarity search.
- Strategy B (AI-Enhanced Retrieval): Expands and cleans up the user’s query before searching, so you get more context-aware hits.

---

## Project walkthrough

Here’s the quick tour of the repo.

### src/ (the engine room)

- data_ingestion.py: Loads a mock set of technical paragraphs and preps them for indexing.
- mock_vertexai.py: Local stand-in for Vertex AI. Uses sentence-transformers for embeddings plus some deterministic logic to mimic a GenerativeModel.
- vector_store.py: FAISS-backed vector search (IndexFlatIP) for fast cosine-sim retrieval.
- retriever.py: The RAGOrchestrator ties everything together—retrieval plus lightweight generation.
- benchmark.py: Runs the A vs. B showdown and writes the markdown report.

### tests/ (quality control)

- test_retriever.py: Pytest suite covering the mocks, the vector store, and the full retrieval pipeline.

### retrieval_benchmark.md (results)

- The final report: metrics, side-by-side results, and notes on swapping in real Vertex AI later.

---

## Getting started

I recommend using a virtual environment to keep things tidy.

### 1) Create and activate a venv

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the project

### 1) Run the benchmark

Generates the comparison and writes the markdown report:

```bash
python -m src.benchmark
```

### 2) Run the tests

Quick health check to make sure everything’s wired up:

```bash
python -m pytest tests/
```

