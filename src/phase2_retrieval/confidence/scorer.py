import os
import json
import time
from utils.logger import log_step

class ConfidenceScorer:
    def _score_with_llm(self, prompt):
        # 1. Check for OpenAI key
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content.strip())
            
        # 2. Check for Google Gemini key
        gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
            response = model.generate_content(prompt)
            return json.loads(response.text.strip())

        raise ValueError("No LLM API keys found (OPENAI_API_KEY or GOOGLE_API_KEY).")

    def calculate_score(self, context, extracted_data, rag_results):
        start_time = time.time()
        log_step("Evaluating confidence score mapping question intent to schema context and rules...")
        # Define LLM Confidence Prompt Template
        prompt = f"""
You are a Data Warehouse NLP Confidence Assessor. Assess the confidence score (from 0 to 100) of mapping the user question to the database schema context and retrieved business rules.

Database context:
- Matched Entities: {context.get("entities", [])}
- Matched Metrics: {context.get("metrics", [])}
- Matched Relationships: {context.get("relationships", [])}

Retrieved Semantic Rules/Glossary chunks:
{[r["chunk"] for r in rag_results] if rag_results else "None"}

Extracted Intent:
- Metrics: {extracted_data.get("metrics", [])}
- Entities: {extracted_data.get("entities", [])}
- Time Grain: {extracted_data.get("time_grain")}
- Filters: {extracted_data.get("filters", [])}

Assess the following:
1. Schema coverage (do the matched entities/metrics cover all tokens in the query?)
2. Ambiguities (is there a missing relationship, or are there multiple matching options?)
3. Business rules constraints (do the retrieved rules contradict or specify extra filters?)

Return JSON in this format:
{{
  "score": 85,
  "reasoning": "Detailed explanation of why this score was assigned."
}}
Return ONLY raw JSON.
"""
        print("\n--- [LLM Confidence Prompt Template] ---")
        print(prompt.strip())
        print("-" * 40)

        try:
            # Attempt LLM Scoring
            result = self._score_with_llm(prompt)
            score = int(result.get("score", 0))
            reasoning = result.get("reasoning", "No reasoning provided.")
            log_step(f"LLM Confidence Engine successfully evaluated score: {score}/100")
            log_step(f"LLM Confidence Reasoning: {reasoning}")
            log_step("Confidence Scoring (LLM) completed", start_time)
            return score
        except Exception as e:
            # Fallback Flow
            log_step(f"LLM Confidence Engine fallback triggered: {str(e)}")
            log_step("Executing local rule-based confidence scorer...")
            
            score = 0
            factors = {}
            
            # 1. Metadata Match (Max 30)
            metadata_score = 0
            if context["entities"]: metadata_score += 15
            if context["metrics"]: metadata_score += 15
            factors["Metadata Match"] = metadata_score
            score += metadata_score
            
            # 2. Relationship Coverage (Max 20)
            rel_score = 0
            if len(context["entities"]) > 1:
                if context["relationships"]:
                    rel_score = 20
                else:
                    rel_score = 5 # Missing expected relationships
            else:
                rel_score = 20 # No complex relationships needed
            factors["Relationship Match"] = rel_score
            score += rel_score
            
            # 3. Vector Similarity / Rules Match (Max 35)
            sim_score = 0
            if rag_results:
                avg_sim = sum([r["final_hybrid_score"] for r in rag_results]) / len(rag_results)
                sim_score = min(35, int(avg_sim * 35))
            factors["Similarity & Rules"] = sim_score
            score += sim_score
            
            # 4. Question Understanding (Max 15)
            understanding_score = 0
            if extracted_data["entities"] or extracted_data["metrics"]:
                 understanding_score = 15
            factors["Question Understanding"] = understanding_score
            score += understanding_score

            log_step(f"Deterministic Fallback Confidence Factors: {factors}")
            log_step(f"Total Confidence Score: {score}/100")
            log_step("Confidence Scoring (Fallback) completed", start_time)
            
            return score

class ClarificationEngine:
    def check_needs_clarification(self, score, extracted_data, context):
        start_time = time.time()
        log_step("Checking if confidence score requires user clarification...")
        
        clarified = False
        if score < 70:
            if not context["metrics"] or not context["entities"]:
                log_step("Confidence is below 70. Clarification Required:")
                if not context["metrics"]:
                    log_step("- Which specific metric are you looking for? (e.g., Gross Revenue, Net Revenue, Profit?)")
                if not context["entities"]:
                    log_step("- By which dimension would you like to see this? (e.g., by Customer, by Region?)")
                clarified = True
        
        log_step(f"Clarification Engine completed. Needs Clarification: {clarified}", start_time)
        return clarified

