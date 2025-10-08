from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class CognitronRetriever:
    def __init__(self, docs):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.docs = docs
        print("Generating embeddings for documents...")
        self.embeddings = self.model.encode(docs, convert_to_numpy=True)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        print(f"Cognitron-RAG-Engine initialized with {len(docs)} documents.")

    def get_relevant_context(self, query, top_k=3):
        """Retrieve the most relevant document chunks for a given query."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        relevant_docs = [self.docs[i] for i in indices[0]]
        return relevant_docs, distances[0]
