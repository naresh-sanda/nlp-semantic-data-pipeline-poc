import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import time

# Add project root and src to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))


def initialize_knowledge_base(embedder):
    """Loads semantic business rules and glossary into ChromaDB."""
    from utils.logger import log_step
    from src.phase1_ingestion.chunking.chunker import Chunker
    
    init_start = time.time()
    log_step("Starting Phase 1 Initialization (knowledge base loading)...")

    rules_path = os.path.join(project_root, "metadata", "semantic_business_rules.txt")
    glossary_path = os.path.join(project_root, "metadata", "semantic_glossary.txt")

    chunker = Chunker(strategy="metadata-aware")
    chunks = []

    if os.path.exists(rules_path):
        read_start = time.time()
        with open(rules_path, 'r', encoding='utf-8') as f:
            content = f.read()
        log_step(f"Read {rules_path} ({len(content)} chars)", read_start)
        chunks.extend(chunker.chunk_text(content))
    else:
        log_step(f"Warning: {rules_path} not found.")

    if os.path.exists(glossary_path):
        read_start = time.time()
        with open(glossary_path, 'r', encoding='utf-8') as f:
            content = f.read()
        log_step(f"Read {glossary_path} ({len(content)} chars)", read_start)
        chunks.extend(chunker.chunk_text(content))
    else:
        log_step(f"Warning: {glossary_path} not found.")

    if chunks:
        # Prevent re-adding if already populated (simplified check)
        stats_start = time.time()
        stats = embedder.get_collection_stats()
        log_step("Checked ChromaDB collection stats", stats_start)
        if stats == 0:
            embedder.add_chunks(chunks)
        else:
            log_step("ChromaDB collection already populated, skipping indexing.")
            
    log_step("Completed Phase 1 Initialization (knowledge base loading)", init_start)


def main():
    from utils.logger import log_step
    total_start = time.time()
    # Deferred imports - torch/chromadb only loaded when main() runs
    from src.phase1_ingestion.loaders.metadata_loader import setup_database
    from src.phase1_ingestion.embeddings.embedder import Embedder

    log_step("=" * 60)
    log_step("NLP Semantic Data Pipeline Builder - Phase 1 Ingestion Starting")
    log_step("=" * 60)

    db_path = os.path.join(project_root, "db", "metadata.db")
    metadata_dir = os.path.join(project_root, "metadata")
    vector_dir = os.path.join(project_root, "vector_store")

    # Ensure folders exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # 1. Setup SQLite
    log_step("Step 1: Setting up SQLite Metadata Tables...")
    step1_start = time.time()
    setup_database(db_path, metadata_dir)
    log_step("SQLite Metadata Setup complete", step1_start)

    # 2. Setup ChromaDB
    log_step("Step 2: Splitting Documents & Indexing Vector Embeddings in ChromaDB...")
    step2_start = time.time()
    embedder = Embedder(vector_dir)
    initialize_knowledge_base(embedder)
    log_step("ChromaDB Embeddings Ingestion complete", step2_start)

    log_step("PHASE 1 INGESTION COMPLETE. STORAGE LAYERS READY FOR RETRIEVAL.", total_start)


if __name__ == "__main__":
    main()
