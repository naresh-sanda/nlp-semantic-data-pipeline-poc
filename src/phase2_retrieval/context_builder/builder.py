import json
import time
from utils.logger import log_step

class ContextBuilder:
    def build_context(self, lookup_results, rag_results):
        """Merges metadata lookup results with RAG results into a single context object."""
        start_time = time.time()
        log_step("Assembling LLM Prompt Context by merging metadata schemas and retrieved rules...")
        
        context = {
            "entities": lookup_results.get("matched_entities", []),
            "metrics": lookup_results.get("matched_metrics", []),
            "relationships": lookup_results.get("relationships", []),
            "business_rules": [res["chunk"] for res in rag_results]
        }
        
        log_step("Unified Context Object:")
        print(json.dumps(context, indent=2))
        
        log_step("Context Construction completed", start_time)
        return context
