import pytest
import numpy as np
from src.mock_vertexai import TextEmbeddingModel, GenerativeModel
from src.vector_store import LocalVectorStore
from src.retriever import RAGOrchestrator

def test_mock_embedding_model():
    model = TextEmbeddingModel()
    texts = ["Hello world", "Testing embeddings"]
    embeddings = model.get_embeddings(texts)
    
    assert len(embeddings) == 2
    assert hasattr(embeddings[0], 'values')
    assert len(embeddings[0].values) == 384 # all-MiniLM-L6-v2 dimension

def test_mock_generative_model():
    model = GenerativeModel()
    prompt = "How does the system handle peak load?"
    response = model.generate_content(prompt)
    
    assert hasattr(response, 'text')
    assert "load balancing" in response.text

def test_local_vector_store():
    store = LocalVectorStore(embedding_dim=4)
    # create some dummy normalized embeddings
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ]
    docs = ["doc1", "doc2"]
    
    store.add_embeddings(embeddings, docs)
    
    query = [1.0, 0.0, 0.0, 0.0]
    results, scores = store.search(query, top_k=1)
    
    assert len(results) == 1
    assert results[0] == "doc1"
    # Cosine similarity of identical normalized vectors is 1.0
    assert np.isclose(scores[0], 1.0, atol=1e-5)

def test_rag_orchestrator_integration():
    orchestrator = RAGOrchestrator()
    orchestrator.ingest()
    
    results_a = orchestrator.retrieve_strategy_a("What is caching?", top_k=2)
    assert len(results_a) == 2
    assert any("Caching" in doc for doc in results_a)
