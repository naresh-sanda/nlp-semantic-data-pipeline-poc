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


def main():
    # Deferred imports - torch/chromadb are only loaded when main() runs
    from src.phase2_retrieval.analyzer.nlp_analyzer import NLPAnalyzer
    from src.phase2_retrieval.metadata_lookup.lookup_engine import LookupEngine
    from src.phase2_retrieval.retrieval.rag_retriever import RAGRetriever
    from src.phase2_retrieval.context_builder.builder import ContextBuilder
    from src.phase2_retrieval.sql_generator.generator import SQLGenerator
    from src.phase2_retrieval.confidence.scorer import ConfidenceScorer, ClarificationEngine

    db_path = "db/metadata.db"
    vector_dir = "vector_store"

    # Check if Phase 1 was run
    chroma_db_file = os.path.join(vector_dir, "chroma.sqlite3")
    if not os.path.exists(db_path) or not os.path.exists(chroma_db_file):
        print("!" * 60)
        print("ERROR: Storage indexes not found!")
        print("Please run Phase 1 Ingestion first to build databases:")
        print("python run_phase1_ingestion.py")
        print("!" * 60)
        return

    print("=" * 60)
    print("NLP Semantic Data Pipeline Builder - Phase 2 Retrieval")
    print("=" * 60)

    # Initialize Engines
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

        # 1. Analyze Question Intent
        extracted_data = analyzer.analyze(q)

        # 2. Lookup Metadata Schema Mapping
        lookup_results = lookup_engine.lookup(extracted_data)

        # 3. Retrieve RAG Rules (Vector Search)
        rag_results = retriever.retrieve(q)

        # 4. Build Prompt Context Object
        context = context_builder.build_context(lookup_results, rag_results)

        # 5. Generate Target SQL Query (LLM / fallback templates)
        sql_generator.generate(context, extracted_data, q)

        # 6. Evaluate Confidence Score (Gatekeeper)
        score = confidence_scorer.calculate_score(context, extracted_data, rag_results)

        # 7. Check if clarification is required
        clarification_engine.check_needs_clarification(score, extracted_data, context)

    print("\n" + "=" * 60)
    print("PHASE 2 RETRIEVAL DEMO RUN COMPLETE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
