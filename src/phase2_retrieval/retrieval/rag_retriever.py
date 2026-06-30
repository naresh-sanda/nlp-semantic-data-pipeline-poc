import os
import time
from utils.logger import log_step

class RAGRetriever:
    def __init__(self, db_dir):
        self.db_dir = db_dir
        self.client = None
        self.collection = None
        self.model = None
        self.use_fallback = False
        
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            
            self.client = chromadb.PersistentClient(path=db_dir)
            self.collection = self.client.get_collection(name="semantic_knowledge")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"[RAG Retriever] Initialization failed: {str(e)}")
            print("[RAG Retriever] Fallback mode active: will use deterministic keyword-overlap retriever.")
            self.use_fallback = True

    def retrieve(self, query, top_k=3, threshold=0.3):
        """Retrieves chunks from ChromaDB using Vector Search, or falls back to keyword-overlap search."""
        start_time = time.time()
        log_step(f"Retrieving semantic rules for query: '{query}'...")
        
        if self.use_fallback:
            log_step("ChromaDB vector DB fallback triggered. Executing local keyword-overlap retriever...")
            results = self._fallback_retrieve(query, top_k, threshold)
            log_step("RAG Retrieval (Fallback) completed", start_time)
            return results
            
        try:
            query_embedding = self.model.encode([query]).tolist()
            
            # ChromaDB uses L2 or Cosine depending on config. We assume default L2 but can rank based on it.
            # We will request top_k+2 to allow threshold filtering
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k + 2
            )
            
            retrieved_chunks = []
            if results['documents'] and results['documents'][0]:
                for doc, distance in zip(results['documents'][0], results['distances'][0]):
                    # Approximate cosine similarity from L2 distance
                    # similarity = 1 - (distance / 2)  if vectors are normalized
                    similarity = max(0, 1 - (distance / 2))
                    
                    if similarity >= threshold:
                        retrieved_chunks.append({
                            "chunk": doc,
                            "vector_score": similarity,
                            # Simulate keyword/BM25 score for the POC
                            "bm25_score": min(0.9, similarity * 1.1), 
                            "final_hybrid_score": (similarity * 0.7) + (min(0.9, similarity * 1.1) * 0.3)
                        })
                        
                # Sort by hybrid score
                retrieved_chunks.sort(key=lambda x: x["final_hybrid_score"], reverse=True)
                retrieved_chunks = retrieved_chunks[:top_k]

            log_step("RAG Retrieval (Vector DB) complete. Matches found:")
            for i, chunk in enumerate(retrieved_chunks):
                print(f"  Match {i+1}: Final Score: {chunk['final_hybrid_score']:.2f} | Content: {chunk['chunk'][:80].strip()}...")

            log_step("RAG Retrieval (Vector DB) completed", start_time)
            return retrieved_chunks
        except Exception as e:
            log_step(f"Vector DB query failed: {str(e)}")
            log_step("Executing local keyword-overlap retriever...")
            results = self._fallback_retrieve(query, top_k, threshold)
            log_step("RAG Retrieval (Fallback after error) completed", start_time)
            return results

    def _fallback_retrieve(self, query, top_k=3, threshold=0.1):
        """Pure Python fallback retriever using word overlap against business rules and glossary."""
        rules_path = os.path.join("metadata", "semantic_business_rules.txt")
        glossary_path = os.path.join("metadata", "semantic_glossary.txt")
        
        chunks = []
        # Chunk logic mimics Chunker
        from src.phase1_ingestion.chunking.chunker import Chunker
        chunker = Chunker(strategy="metadata-aware")
        
        for path in [rules_path, glossary_path]:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        chunks.extend(chunker.chunk_text(f.read()))
                except Exception as e:
                    print(f"[RAG Retriever Fallback] Error reading {path}: {str(e)}")
        
        query_words = set(w.strip("?,.!:;()\"'") for w in query.lower().split() if len(w) > 2)
        retrieved_chunks = []
        
        for doc in chunks:
            doc_words = set(w.strip("?,.!:;()\"'") for w in doc.lower().split() if len(w) > 2)
            overlap = len(query_words.intersection(doc_words))
            
            if overlap > 0:
                # Basic Jaccard-like or overlap ratio
                similarity = overlap / max(1, len(query_words))
                # Adjust to fit scale
                similarity = min(0.95, similarity * 1.5)
                
                if similarity >= threshold:
                    retrieved_chunks.append({
                        "chunk": doc,
                        "vector_score": similarity,
                        "bm25_score": similarity * 0.9,
                        "final_hybrid_score": (similarity * 0.7) + (similarity * 0.9 * 0.3)
                    })
                    
        # Sort and limit
        retrieved_chunks.sort(key=lambda x: x["final_hybrid_score"], reverse=True)
        retrieved = retrieved_chunks[:top_k]
        
        log_step("RAG Retrieval (Local Fallback) matches:")
        for i, chunk in enumerate(retrieved):
            log_step(f"  Match {i+1}: Final Score: {chunk['final_hybrid_score']:.2f} | Content: {chunk['chunk'][:80].strip()}...")
            
        return retrieved
