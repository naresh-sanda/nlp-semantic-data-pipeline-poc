import json

class ContextBuilder:
    def build_context(self, lookup_results, rag_results):
        """Merges metadata lookup results with RAG results into a single context object."""
        
        context = {
            "entities": lookup_results.get("matched_entities", []),
            "metrics": lookup_results.get("matched_metrics", []),
            "relationships": lookup_results.get("relationships", []),
            "business_rules": [res["chunk"] for res in rag_results]
        }
        
        print("\n--- Built Context ---")
        print(json.dumps(context, indent=2))
        
        return context
