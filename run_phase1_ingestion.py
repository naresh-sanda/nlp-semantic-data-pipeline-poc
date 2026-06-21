import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys

# Add project root and src to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))


def initialize_knowledge_base(embedder):
    """Loads semantic business rules and glossary into ChromaDB."""
    from src.phase1_ingestion.chunking.chunker import Chunker

    rules_path = os.path.join("metadata", "semantic_business_rules.txt")
    glossary_path = os.path.join("metadata", "semantic_glossary.txt")

    chunker = Chunker(strategy="metadata-aware")
    chunks = []

    if os.path.exists(rules_path):
        with open(rules_path, 'r', encoding='utf-8') as f:
            chunks.extend(chunker.chunk_text(f.read()))

    if os.path.exists(glossary_path):
        with open(glossary_path, 'r', encoding='utf-8') as f:
            chunks.extend(chunker.chunk_text(f.read()))

    if chunks:
        # Prevent re-adding if already populated (simplified check)
        if embedder.get_collection_stats() == 0:
            embedder.add_chunks(chunks)
        else:
            print("ChromaDB already populated.")


def main():
    # Deferred imports - torch/chromadb only loaded when main() runs
    from src.phase1_ingestion.loaders.metadata_loader import setup_database
    from src.phase1_ingestion.embeddings.embedder import Embedder

    print("=" * 60)
    print("NLP Semantic Data Pipeline Builder - Phase 1 Ingestion")
    print("=" * 60)

    db_path = "db/metadata.db"
    metadata_dir = "metadata"
    vector_dir = "vector_store"

    # Ensure folders exist
    os.makedirs("db", exist_ok=True)

    # 1. Setup SQLite
    print("\n--- [Phase 1] Setting up SQLite Metadata Tables ---")
    setup_database(db_path, metadata_dir)
    print("[SQLite] Metadata setup complete.")

    # 2. Setup ChromaDB
    print("\n--- [Phase 1] Splitting Documents & Indexing Vector Embeddings ---")
    embedder = Embedder(vector_dir)
    initialize_knowledge_base(embedder)
    print("[ChromaDB] Embeddings Ingestion complete.")

    print("\n" + "=" * 60)
    print("PHASE 1 INGESTION COMPLETE. STORAGE LAYERS READY FOR RETRIEVAL.")
    print("=" * 60)


if __name__ == "__main__":
    main()
