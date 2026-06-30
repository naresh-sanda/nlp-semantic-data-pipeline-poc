import re
import os
import json
import time
from utils.logger import log_step

class NLPAnalyzer:
    def __init__(self):
        # Basic lists of known keywords for the POC
        self.known_metrics = ["revenue", "profit", "margin", "orders", "customers", "sales", "aov"]
        self.known_entities = ["customer", "region", "product", "supplier", "warehouse"]
        self.known_time_grains = ["monthly", "daily", "yearly", "weekly"]

    def _analyze_with_llm(self, question, prompt):
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

    def analyze(self, question):
        """Extracts entities, metrics, dimensions, and time periods from a question."""
        start_time = time.time()
        log_step(f"Starting NLP Analysis for query: '{question}'...")
        
        # Define LLM Prompt Template
        prompt = f"""
You are a Data Warehouse NLP Planner. Extract metrics, entities, time grains, and limit filters from the user question.

Allowed metrics: {self.known_metrics}
Allowed entities: {self.known_entities}
Allowed time grains: {self.known_time_grains}

User Question: "{question}"

Return JSON in this format:
{{
  "metrics": ["metric_name"],
  "entities": ["entity_name"],
  "time_grain": "monthly" or null,
  "filters": [{{"type": "limit", "value": 10}}] or []
}}
Return ONLY raw JSON.
"""
        log_step("Prepared LLM Query Planner Prompt Template:")
        print(prompt.strip())
        print("-" * 43)

        try:
            # Attempt Production Flow
            extracted = self._analyze_with_llm(question, prompt)
            log_step("Successfully extracted intent using LLM.")
        except Exception as e:
            # Fallback Flow
            log_step(f"LLM intent extraction fallback triggered: {str(e)}")
            log_step("Executing local deterministic intent parser...")
            
            question_lower = question.lower()
            extracted = {
                "metrics": [],
                "entities": [],
                "time_grain": None,
                "filters": []
            }

            # Extract Metrics
            for metric in self.known_metrics:
                if metric in question_lower:
                    extracted["metrics"].append(metric)

            # Extract Entities/Dimensions
            for entity in self.known_entities:
                if entity in question_lower:
                    extracted["entities"].append(entity)

            # Extract Time Grain
            for time_grain in self.known_time_grains:
                if time_grain in question_lower:
                    extracted["time_grain"] = time_grain
                    
            # Basic filter extraction (e.g. "top 10")
            top_match = re.search(r'top (\d+)', question_lower)
            if top_match:
                extracted["filters"].append({"type": "limit", "value": int(top_match.group(1))})

        log_step(f"NLP intent extraction completed. Extracted: {extracted}", start_time)
        return extracted
