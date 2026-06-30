import os
import time
from utils.logger import log_step

class Embedder:
    def __init__(self, db_dir):
        self.db_dir = db_dir
        self.client = None
        self.collection = None
        self.model = None
        self.use_fallback = False
        
        try:
            # pyrefly: ignore [missing-import]
            import chromadb
            # pyrefly: ignore [missing-import]
            from sentence_transformers import SentenceTransformer
            
            os.makedirs(db_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=db_dir)
            self.collection = self.client.get_or_create_collection(name="semantic_knowledge")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            # Support both old and new sentence-transformers API
            try:
                dim = self.model.get_sentence_embedding_dimension()
            except AttributeError:
                dim = self.model[0].get_word_embedding_dimension() if hasattr(self.model, '__getitem__') else 384
            log_step(f"Embedder initialized. Embedding dimensions: {dim}")
        except Exception as e:
            log_step(f"Initialization failed: {str(e)}")
            log_step("Fallback mode active: will skip actual vector DB inserts.")
            self.use_fallback = True

    def add_chunks(self, chunks, metadatas=None):
        if not chunks:
            return
            
        start_time = time.time()
        log_step(f"Indexing {len(chunks)} chunks to vector DB...")
        
        if self.use_fallback:
            log_step(f"Simulating chunk storage for {len(chunks)} chunks (Fallback Mode)", start_time)
            return
            
        try:
            encode_start = time.time()
            embeddings = self.model.encode(chunks).tolist()
            log_step(f"Generated {len(chunks)} embeddings", encode_start)
            
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            
            if metadatas is None:
                metadatas = [{"source": "unknown"} for _ in chunks]

            upsert_start = time.time()
            # Use upsert to handle re-ingestion without duplicate ID errors
            self.collection.upsert(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            log_step(f"Added/Updated {len(chunks)} chunks in ChromaDB collection", upsert_start)
            log_step("Vector DB Ingestion completed", start_time)
        except Exception as e:
            log_step(f"Failed to add chunks to ChromaDB: {str(e)}")
            log_step("Fallback active: simulated insertion completed", start_time)

    def get_collection_stats(self):
        if self.use_fallback or not self.collection:
            print("ChromaDB Collection Size: 0 documents (Fallback Mode)")
            return 0
        try:
            count = self.collection.count()
            print(f"ChromaDB Collection Size: {count} documents")
            return count
        except Exception as e:
            print(f"[Embedder] Failed to get stats: {str(e)}")
            return 0

