from src.mock_vertexai import MockTextEmbeddingModel, MockGenerativeModel
from src.vector_store import SimpleVectorStore
from src.data_ingestion import DataIngester

class RAGPipeline:
    def __init__(self):
        # setting up our local mocks
        print("Initializing RAG Pipeline components...")
        self.embedder = MockTextEmbeddingModel.from_pretrained("textembedding-gecko")
        self.llm = MockGenerativeModel("gemini-1.5-pro-preview-0409")
        
        # 384 is what all-MiniLM-L6-v2 outputs
        self.store = SimpleVectorStore(dim=384)
        self.ingester = DataIngester()

    def build_index(self):
        print("Starting index build process.")
        raw_docs = self.ingester.load_data()
        
        # embed everything
        mock_embs = self.embedder.get_embeddings(raw_docs)
        embs_list = [e.values for e in mock_embs]
        
        # shove it into FAISS
        self.store.add(embs_list, raw_docs)
        print("Successfully built FAISS index with ingested data.")

    def search_vanilla(self, query, top_k=3):
        # Strategy A: standard search
        print(f"Executing Strategy A (Vanilla Search) for: '{query}'")
        q_emb = self.embedder.get_embeddings([query])[0].values
        hits, _ = self.store.search(q_emb, k=top_k)
        return hits

    def search_expanded(self, query, top_k=3):
        # Strategy B: ask LLM to pull out keywords first
        print(f"Executing Strategy B (Expanded Search) for: '{query}'")
        prompt = f"Rewrite this query to pull out key technical terms for search: '{query}'"
        
        resp = self.llm.generate_content(prompt)
        expanded = resp.text
        print(f"LLM expanded query to: '{expanded}'")
        
        # use the new expanded query to search
        q_emb = self.embedder.get_embeddings([expanded])[0].values
        hits, _ = self.store.search(q_emb, k=top_k)
        
        return hits, expanded
