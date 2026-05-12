import os
from src.retriever import RAGPipeline

def run_it():
    rag = RAGPipeline()
    print("Loading and indexing data...")
    rag.build_index()
    print("Done indexing.\n")

    test_queries = [
        "How does the system handle peak load?",
        "What is the best way to speed up database queries?",
        "How do microservices communicate asynchronously?"
    ]

    results = []

    for q in test_queries:
        print(f"Running queries for: '{q}'")
        
        # basic search
        vanilla_hits = rag.search_vanilla(q, top_k=3)
        
        # AI expanded search
        expanded_hits, new_q = rag.search_expanded(q, top_k=3)

        results.append({
            "original_query": q,
            "ai_query": new_q,
            "strategy_a": vanilla_hits,
            "strategy_b": expanded_hits
        })

    # write it all out to markdown
    out_file = "retrieval_benchmark.md"
    with open(out_file, "w") as f:
        f.write("# Retrieval Benchmark Report\n\n")
        f.write("## Strategy A (Raw Search) vs. Strategy B (Query Expansion)\n\n")

        for idx, r in enumerate(results, start=1):
            f.write(f"### Test {idx}: {r['original_query']}\n")
            f.write(f"**Expanded by LLM**: {r['ai_query']}\n\n")
            
            f.write("| Rank | Strategy A | Strategy B |\n")
            f.write("| --- | --- | --- |\n")
            
            for i in range(3):
                # safe fallback if we get less than 3 hits
                hit_a = r['strategy_a'][i] if i < len(r['strategy_a']) else "N/A"
                hit_b = r['strategy_b'][i] if i < len(r['strategy_b']) else "N/A"
                
                # truncate so the table doesn't look terrible
                trunc_a = hit_a[:100] + "..." if len(hit_a) > 100 else hit_a
                trunc_b = hit_b[:100] + "..." if len(hit_b) > 100 else hit_b
                
                f.write(f"| {i+1} | {trunc_a} | {trunc_b} |\n")
            f.write("\n")

        # add some notes about the tech decisions
        f.write("## Notes\n\n")
        f.write("### Why Cosine Similarity?\n")
        f.write("I went with Cosine Similarity over Euclidean. Since we're dealing with text embeddings (from sentence-transformers/Vertex), the length or magnitude of the vector shouldn't skew the results—we care about the angle (the semantics). ")
        f.write("To make FAISS do this, I'm normalizing the vectors first and then running an Inner Product search (`IndexFlatIP`). Same math, faster search.\n\n")

        f.write("### Moving to Vertex AI\n")
        f.write("If we pushed this to prod, here's how we'd swap out the local setup:\n")
        f.write("1. **Embeddings**: Run the actual `vertexai.language_models.TextEmbeddingModel` against the docs, dump to JSONL, and upload to GCS.\n")
        f.write("2. **Vector DB**: Spin up a `MatchingEngineIndex` in GCP. `DOT_PRODUCT_DISTANCE` works fine if we normalize, or just use Cosine natively.\n")
        f.write("3. **Deployment**: Deploy it to an `IndexEndpoint`.\n")
        f.write("4. **App Code**: Swap out the `SimpleVectorStore` with an SDK call to `aiplatform.MatchingEngineIndexEndpoint.match()`. Then fetch the actual text payloads from something like Cloud SQL or Firestore based on the returned IDs.\n")

    print(f"Finished! Check out the results in {out_file}")

if __name__ == "__main__":
    run_it()
