import os
# Set thread limits BEFORE importing torch/numpy
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


def main():
    from utils.logger import log_step
    demo_start = time.time()
    log_step("=" * 70)
    log_step("DATA PIPELINE - NLP POC DEMO RUNNER")
    log_step("=" * 70)

    # ── Phase 1: Ingestion ────────────────────────────────────────────────
    log_step(">>> STARTING PHASE 1: INGESTION...")
    phase1_start = time.time()
    from src.phase1_ingestion.loaders.metadata_loader import setup_database
    from src.phase1_ingestion.chunking.chunker import Chunker
    from src.phase1_ingestion.embeddings.embedder import Embedder

    db_path = os.path.join(project_root, "db", "metadata.db")
    metadata_dir = os.path.join(project_root, "metadata")
    vector_dir = os.path.join(project_root, "vector_store")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    log_step("Step 1: Setting up SQLite Metadata Tables...")
    step1_start = time.time()
    setup_database(db_path, metadata_dir)
    log_step("SQLite Metadata Setup complete", step1_start)

    log_step("Step 2: Splitting Documents & Indexing Vector Embeddings...")
    step2_start = time.time()
    embedder = Embedder(vector_dir)
    rules_path = os.path.join(metadata_dir, "semantic_business_rules.txt")
    glossary_path = os.path.join(metadata_dir, "semantic_glossary.txt")
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
        embedder.add_chunks(chunks)
    log_step("ChromaDB Embeddings Ingestion complete", step2_start)
    log_step("=" * 70)
    log_step("PHASE 1 INGESTION COMPLETE. STORAGE LAYERS READY FOR RETRIEVAL", phase1_start)
    log_step("=" * 70)

    # ── Phase 2: Retrieval ────────────────────────────────────────────────
    log_step(">>> STARTING PHASE 2: RETRIEVAL...")
    phase2_start = time.time()
    from src.phase2_retrieval.analyzer.nlp_analyzer import NLPAnalyzer
    from src.phase2_retrieval.metadata_lookup.lookup_engine import LookupEngine
    from src.phase2_retrieval.retrieval.rag_retriever import RAGRetriever
    from src.phase2_retrieval.context_builder.builder import ContextBuilder
    from src.phase2_retrieval.sql_generator.generator import SQLGenerator
    from src.phase2_retrieval.confidence.scorer import ConfidenceScorer, ClarificationEngine

    if not os.path.exists(db_path) or not os.path.exists(vector_dir):
        log_step("ERROR: Storage indexes not found! Phase 1 ingestion may have failed.")
        return

    log_step("=" * 60)
    log_step("NLP Semantic Data Pipeline Builder - Phase 2 Retrieval")
    log_step("=" * 60)

    analyzer = NLPAnalyzer()
    lookup_engine = LookupEngine(db_path)
    retriever = RAGRetriever(vector_dir)
    context_builder = ContextBuilder()
    sql_generator = SQLGenerator()
    confidence_scorer = ConfidenceScorer()
    clarification_engine = ClarificationEngine()

    test_questions = [
        "Show monthly revenue by customer",
        "Top 10 customers by profit",
        "Revenue by region",
        "Product margin analysis",
        "Orders by supplier",
        "Customer growth trend",
        "Sales by warehouse"
    ]

    for q in test_questions:
        query_start = time.time()
        log_step("=" * 60)
        log_step(f"PROCESSING RUNTIME QUERY: '{q}'")
        log_step("=" * 60)
        extracted_data = analyzer.analyze(q)
        lookup_results = lookup_engine.lookup(extracted_data)
        rag_results = retriever.retrieve(q)
        context = context_builder.build_context(lookup_results, rag_results)
        sql_generator.generate(context, extracted_data, q)
        score = confidence_scorer.calculate_score(context, extracted_data, rag_results)
        clarification_engine.check_needs_clarification(score, extracted_data, context)
        log_step(f"[QUERY LIFECYCLE] Finished processing query: '{q}'", query_start)

    log_step("=" * 70)
    log_step(f"DEMO PIPELINE COMPLETE (Phase 2 took {time.time() - phase2_start:.4f}s)", demo_start)
    log_step("=" * 70)


if __name__ == "__main__":
    main()
