import chromadb
from sentence_transformers import SentenceTransformer
import chromadb


from typing import List

class VectorStore:
    def __init__(self):
        self.client = chromadb.Client()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection = self.client.get_or_create_collection(
            name='papers',
            metadata={'hnsw:space': 'cosine'}
        )
        try:
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except Exception:
            pass

    def add_chunks(self, chunks: List[str], source: str):
        embeddings = self.model.encode(chunks).tolist()
        ids = [f'{source}_{i}' for i in range(len(chunks))]
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{'source': source, 'chunk_id': i} for i in range(len(chunks))]
        )

    def search(self, query: str, n_results: int = 5) -> List[str]:
        embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=n_results
        )
        if results['documents'] and len(results['documents'][0]) > 0:
            return results['documents'][0]
        return []

    def clear(self):
        try:
            self.client.delete_collection('papers')
        except:
            pass
        self.collection = self.client.get_or_create_collection('papers')
