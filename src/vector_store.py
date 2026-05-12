import faiss
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SimpleVectorStore:
    def __init__(self, dim=384):
        # using inner product + normalized vectors to get cosine similarity
        logger.info(f"Setting up FAISS IndexFlatIP with dimension {dim}")
        self.dim = dim
        self.index = faiss.IndexFlatIP(self.dim)
        self.docs = []

    def _normalize(self, vecs):
        vecs = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        # prevent div by zero
        norms = np.where(norms == 0, 1e-10, norms)
        return vecs / norms

    def add(self, embeddings, text_docs):
        # normalize before adding to faiss
        logger.info(f"Adding {len(text_docs)} documents to the vector store.")
        norm_embs = self._normalize(embeddings)
        self.index.add(norm_embs)
        self.docs.extend(text_docs)

    def search(self, query_emb, k=3):
        if not self.docs:
            logger.error("Search called on an empty vector store.")
            return [], []

        logger.info(f"Searching FAISS index for top {k} results.")
        q_norm = self._normalize([query_emb])
        dists, idxs = self.index.search(q_norm, k)
        
        res = []
        scores = []
        for j, index in enumerate(idxs[0]):
            if index != -1 and index < len(self.docs):
                res.append(self.docs[index])
                scores.append(dists[0][j])
                
        return res, scores
