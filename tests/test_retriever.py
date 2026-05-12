import pytest
import numpy as np
from src.mock_vertexai import MockTextEmbeddingModel, MockGenerativeModel
from src.vector_store import SimpleVectorStore
from src.retriever import RAGPipeline

def test_embedder_mock():
    embedder = MockTextEmbeddingModel()
    res = embedder.get_embeddings(["test 1", "test 2"])
    
    assert len(res) == 2
    assert hasattr(res[0], 'values')
    assert len(res[0].values) == 384 # making sure the dim matches what we expect from MiniLM

def test_llm_mock():
    llm = MockGenerativeModel()
    resp = llm.generate_content("How does the system handle peak load?")
    
    # should pull from the hardcoded cheat sheet
    assert "load balancing" in resp.text

def test_faiss_store():
    store = SimpleVectorStore(dim=4)
    dummy_embs = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ]
    
    store.add(dummy_embs, ["doc_a", "doc_b"])
    
    hits, scores = store.search([1.0, 0.0, 0.0, 0.0], k=1)
    
    assert len(hits) == 1
    assert hits[0] == "doc_a"
    assert np.isclose(scores[0], 1.0, atol=1e-5) # cosine sim of identical vectors is 1

def test_full_pipeline():
    rag = RAGPipeline()
    rag.build_index()
    
    hits = rag.search_vanilla("What is caching?", top_k=2)
    assert len(hits) == 2
    # quick check to make sure it's returning something relevant
    assert any("Caching" in h for h in hits)
