import os

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
            print(f"Embedder initialized. Embedding dimensions: {dim}")
        except Exception as e:
            print(f"[Embedder] Initialization failed: {str(e)}")
            print("[Embedder] Fallback mode active: will skip actual vector DB inserts.")
            self.use_fallback = True

    def add_chunks(self, chunks, metadatas=None):
        if not chunks:
            return
            
        if self.use_fallback:
            print(f"[Embedder Fallback] Simulating chunk storage for {len(chunks)} chunks.")
            return
            
        try:
            embeddings = self.model.encode(chunks).tolist()
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            
            if metadatas is None:
                metadatas = [{"source": "unknown"} for _ in chunks]

            # Use upsert to handle re-ingestion without duplicate ID errors
            self.collection.upsert(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added/Updated {len(chunks)} chunks in ChromaDB collection.")
        except Exception as e:
            print(f"[Embedder] Failed to add chunks to ChromaDB: {str(e)}")
            print("[Embedder] Fallback active: simulated insertion.")

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

