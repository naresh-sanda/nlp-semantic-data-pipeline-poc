import os

class SQLGenerator:
    def _generate_with_llm(self, prompt):
        # 1. Check for OpenAI key
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            sql_out = response.choices[0].message.content.strip()
            # Clean up markdown if model outputs it
            if sql_out.startswith("```sql"):
                sql_out = sql_out[6:]
            if sql_out.endswith("```"):
                sql_out = sql_out[:-3]
            return sql_out.strip()
            
        # 2. Check for Google Gemini key
        gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            sql_out = response.text.strip()
            # Clean up markdown if model outputs it
            if sql_out.startswith("```sql"):
                sql_out = sql_out[6:]
            if sql_out.endswith("```"):
                sql_out = sql_out[:-3]
            return sql_out.strip()

        raise ValueError("No LLM API keys found (OPENAI_API_KEY or GOOGLE_API_KEY).")

    def generate(self, context, extracted_data, question=None):
        """Generates ANSI SQL using LLM if keys are available, falling back to deterministic templates."""
        
        # Define LLM Prompt Template for SQL Generation
        prompt = f"""
You are a SQL Expert for an ANSI-compliant Data Warehouse. Generate an ANSI SQL query based on the database metadata context, retrieved business rules, and user question.

Database Context:
- Matched Entities: {context.get("entities", [])}
- Matched Metrics: {context.get("metrics", [])}
- Matched Relationships: {context.get("relationships", [])}

Retrieved Semantic Business Rules:
{context.get("business_rules", [])}

Extracted Intent Details:
- Time Grain: {extracted_data.get("time_grain")}
- Filters/Limits: {extracted_data.get("filters", [])}

Original User Question: "{question or 'Not specified'}"

Ensure that:
1. All join conditions between tables are clearly specified in the WHERE clause or JOIN clauses.
2. GROUP BY columns match selected dimensions.
3. Appropriate aggregation functions (e.g., SUM, COUNT, AVG) are applied.
4. If a time grain is "monthly", group and truncate dates to month (e.g., DATE_TRUNC('month', order_date)).
5. Return ONLY the raw SQL query. Do not wrap in markdown or any other tags.
"""
        print("\n--- [LLM SQL Generator Prompt Template] ---")
        print(prompt.strip())
        print("-" * 43)

        try:
            # Attempt Production Flow
            sql = self._generate_with_llm(prompt)
            print("[LLM Generator] Successfully generated SQL using LLM.")
            print("\n--- Generated SQL ---")
            print(sql)
            return sql
        except Exception as e:
            # Fallback Flow
            print(f"[LLM Generator] Fallback triggered: {str(e)}")
            print("[LLM Generator] Executing local template-based SQL generator...")
            
            metrics = context.get("metrics", [])
            entities = context.get("entities", [])
            
            if not metrics and not entities:
                return "SELECT * FROM dual; -- Could not determine intention"

            select_clause = []
            group_by_clause = []
            from_clause = []
            
            # Simple heuristic mapping for POC
            for e in entities:
                entity_name = e["name"].lower()
                if entity_name == "customer":
                    select_clause.append("c.customer_name")
                    group_by_clause.append("c.customer_name")
                    if "customers c" not in from_clause:
                        from_clause.append("customers c")
                elif entity_name == "region":
                    select_clause.append("r.region_name")
                    group_by_clause.append("r.region_name")
                    if "regions r" not in from_clause:
                        from_clause.append("regions r")
                elif entity_name == "product":
                    select_clause.append("p.product_name")
                    group_by_clause.append("p.product_name")
                    if "products p" not in from_clause:
                        from_clause.append("products p")
                elif entity_name == "supplier":
                    select_clause.append("s.supplier_name")
                    group_by_clause.append("s.supplier_name")
                    if "suppliers s" not in from_clause:
                        from_clause.append("suppliers s")
                elif entity_name == "warehouse":
                    select_clause.append("w.warehouse_name")
                    group_by_clause.append("w.warehouse_name")
                    if "warehouses w" not in from_clause:
                        from_clause.append("warehouses w")

            for m in metrics:
                metric_name = m["name"].lower()
                if metric_name == "revenue":
                    select_clause.append("SUM(o.net_amount) as total_revenue")
                    if "orders o" not in from_clause:
                        from_clause.append("orders o")
                elif metric_name == "profit":
                    select_clause.append("SUM(o.net_amount - o.cost) as total_profit")
                    if "orders o" not in from_clause:
                        from_clause.append("orders o")
                elif metric_name == "margin":
                    select_clause.append("(SUM(o.net_amount - o.cost) / SUM(o.net_amount)) * 100 as profit_margin_pct")
                    if "orders o" not in from_clause:
                        from_clause.append("orders o")
                elif metric_name in ["ordercount", "orders"]:
                    select_clause.append("COUNT(o.order_id) as total_orders")
                    if "orders o" not in from_clause:
                        from_clause.append("orders o")
                elif metric_name in ["customercount", "customers"]:
                    select_clause.append("COUNT(DISTINCT o.customer_id) as total_customers")
                    if "orders o" not in from_clause:
                        from_clause.append("orders o")

            # Time grain
            time_grain = extracted_data.get("time_grain")
            if time_grain == "monthly":
                select_clause.insert(0, "DATE_TRUNC('month', o.order_date) as order_month")
                group_by_clause.insert(0, "DATE_TRUNC('month', o.order_date)")
            
            # Join Resolution logic:
            # Check if we have suppliers s and orders o. If so, and products p is not there, we need to add it to join them.
            has_supplier = any("suppliers" in f for f in from_clause)
            has_orders = any("orders" in f for f in from_clause)
            has_products = any("products" in f for f in from_clause)
            
            if has_supplier and has_orders and not has_products:
                from_clause.append("products p")
                has_products = True

            # Assemble SQL
            sql = f"SELECT {', '.join(select_clause)}\n"
            sql += f"FROM {', '.join(from_clause)}\n"
            
            # Joins (Simplified for POC but now fully functional/generating conditions)
            join_conditions = []
            if len(from_clause) > 1:
                table_keys = [f.split()[1] for f in from_clause] # get aliases: c, r, p, o, s, w
                
                if "c" in table_keys and "o" in table_keys:
                    join_conditions.append("c.customer_id = o.customer_id")
                if "r" in table_keys and "o" in table_keys:
                    join_conditions.append("r.region_id = o.region_id")
                if "p" in table_keys and "o" in table_keys:
                    join_conditions.append("p.product_id = o.product_id")
                if "s" in table_keys and "p" in table_keys:
                    join_conditions.append("s.supplier_id = p.supplier_id")
                if "w" in table_keys and "o" in table_keys:
                    join_conditions.append("w.warehouse_id = o.warehouse_id")

                if join_conditions:
                    sql += f"WHERE {' AND '.join(join_conditions)}\n"
                 
            if group_by_clause:
                sql += f"GROUP BY {', '.join(group_by_clause)}\n"
                
            # Filters (Limit)
            for f in extracted_data.get("filters", []):
                if f["type"] == "limit":
                    if metrics:
                        # Order by the first metric
                        sql += f"ORDER BY 2 DESC\n" 
                    sql += f"LIMIT {f['value']}\n"
                    
            print("\n--- Generated SQL (Deterministic Fallback) ---")
            print(sql)
            
            return sql

