import os
# Set thread limits BEFORE importing torch/numpy
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys

# Add project root and src to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))


def main():
    print("=" * 70)
    print("DATA PIPELINE - NLP POC DEMO RUNNER")
    print("=" * 70)

    # ── Phase 1: Ingestion ────────────────────────────────────────────────
    print("\n>>> STARTING PHASE 1: INGESTION...")
    from src.phase1_ingestion.loaders.metadata_loader import setup_database
    from src.phase1_ingestion.chunking.chunker import Chunker
    from src.phase1_ingestion.embeddings.embedder import Embedder

    db_path = "db/metadata.db"
    metadata_dir = "metadata"
    vector_dir = "vector_store"
    os.makedirs("db", exist_ok=True)

    print("\n--- [Phase 1] Setting up SQLite Metadata Tables ---")
    setup_database(db_path, metadata_dir)
    print("[SQLite] Metadata setup complete.")

    print("\n--- [Phase 1] Splitting Documents & Indexing Vector Embeddings ---")
    embedder = Embedder(vector_dir)
    rules_path = os.path.join(metadata_dir, "semantic_business_rules.txt")
    glossary_path = os.path.join(metadata_dir, "semantic_glossary.txt")
    chunker = Chunker(strategy="metadata-aware")
    chunks = []
    if os.path.exists(rules_path):
        with open(rules_path, 'r', encoding='utf-8') as f:
            chunks.extend(chunker.chunk_text(f.read()))
    if os.path.exists(glossary_path):
        with open(glossary_path, 'r', encoding='utf-8') as f:
            chunks.extend(chunker.chunk_text(f.read()))
    if chunks:
        embedder.add_chunks(chunks)
    print("[ChromaDB] Embeddings Ingestion complete.")
    print("\n" + "=" * 70)
    print("PHASE 1 INGESTION COMPLETE. STORAGE LAYERS READY FOR RETRIEVAL.")
    print("=" * 70)

    # ── Phase 2: Retrieval ────────────────────────────────────────────────
    print("\n>>> STARTING PHASE 2: RETRIEVAL...")
    from src.phase2_retrieval.analyzer.nlp_analyzer import NLPAnalyzer
    from src.phase2_retrieval.metadata_lookup.lookup_engine import LookupEngine
    from src.phase2_retrieval.retrieval.rag_retriever import RAGRetriever
    from src.phase2_retrieval.context_builder.builder import ContextBuilder
    from src.phase2_retrieval.sql_generator.generator import SQLGenerator
    from src.phase2_retrieval.confidence.scorer import ConfidenceScorer, ClarificationEngine

    if not os.path.exists(db_path) or not os.path.exists(vector_dir):
        print("ERROR: Storage indexes not found! Phase 1 ingestion may have failed.")
        return

    print("\n" + "=" * 60)
    print("NLP Semantic Data Pipeline Builder - Phase 2 Retrieval")
    print("=" * 60)

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
        print("\n" + "=" * 60)
        print(f"PROCESSING RUNTIME QUERY: '{q}'")
        print("=" * 60)
        extracted_data = analyzer.analyze(q)
        lookup_results = lookup_engine.lookup(extracted_data)
        rag_results = retriever.retrieve(q)
        context = context_builder.build_context(lookup_results, rag_results)
        sql_generator.generate(context, extracted_data, q)
        score = confidence_scorer.calculate_score(context, extracted_data, rag_results)
        clarification_engine.check_needs_clarification(score, extracted_data, context)

    print("\n" + "=" * 70)
    print("DEMO PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
