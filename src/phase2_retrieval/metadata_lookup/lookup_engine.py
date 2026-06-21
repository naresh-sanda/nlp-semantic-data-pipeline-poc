import sqlite3
import os
import csv

class LookupEngine:
    def __init__(self, db_path):
        self.db_path = db_path

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def lookup(self, extracted_data):
        """Lookup metadata using SQLite, with a fallback to local CSV parsing on failure."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            results = {
                "matched_entities": [],
                "matched_metrics": [],
                "relationships": []
            }

            # Lookup Entities
            for entity_kw in extracted_data.get("entities", []):
                # Using LIKE to find matches in synonyms or entity_name
                cursor.execute("SELECT entity_id, entity_name FROM semantic_entities WHERE LOWER(entity_name) LIKE ? OR LOWER(synonyms) LIKE ?", 
                               (f"%{entity_kw}%", f"%{entity_kw}%"))
                rows = cursor.fetchall()
                for row in rows:
                    results["matched_entities"].append({"id": row[0], "name": row[1]})

            # Lookup Metrics
            for metric_kw in extracted_data.get("metrics", []):
                cursor.execute("SELECT metric_id, metric_name, calculation_type FROM semantic_metrics WHERE LOWER(metric_name) LIKE ? OR LOWER(synonyms) LIKE ?", 
                               (f"%{metric_kw}%", f"%{metric_kw}%"))
                rows = cursor.fetchall()
                for row in rows:
                    results["matched_metrics"].append({"id": row[0], "name": row[1], "calc_type": row[2]})
                    
            # Lookup Relationships between matched entities
            if len(results["matched_entities"]) > 1:
                entity_names = [e["name"] for e in results["matched_entities"]]
                placeholders = ','.join('?' for _ in entity_names)
                query = f"""
                    SELECT source_entity, target_entity, relationship_type 
                    FROM semantic_relationships 
                    WHERE source_entity IN ({placeholders}) AND target_entity IN ({placeholders})
                """
                # Need to pass parameters twice since we check both source and target
                params = entity_names + entity_names
                cursor.execute(query, params)
                rows = cursor.fetchall()
                for row in rows:
                     results["relationships"].append({"source": row[0], "target": row[1], "type": row[2]})
                     
            conn.close()
            print("\n--- Metadata Lookup (SQLite) ---")
            print(f"Results: {results}")
            return results

        except Exception as e:
            print(f"[Lookup Engine] SQLite lookup failed: {str(e)}")
            print("[Lookup Engine] Fallback triggered. Executing local CSV metadata lookup...")
            return self._fallback_lookup(extracted_data)

    def _fallback_lookup(self, extracted_data):
        """Pure Python fallback parsing semantic CSV files directly."""
        results = {
            "matched_entities": [],
            "matched_metrics": [],
            "relationships": []
        }
        
        entities_path = os.path.join("metadata", "semantic_entities.csv")
        metrics_path = os.path.join("metadata", "semantic_metrics.csv")
        relationships_path = os.path.join("metadata", "semantic_relationships.csv")
        
        # 1. Entities Lookup
        if os.path.exists(entities_path):
            try:
                with open(entities_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        for entity_kw in extracted_data.get("entities", []):
                            kw_l = entity_kw.lower()
                            if kw_l in row["entity_name"].lower() or kw_l in row.get("synonyms", "").lower():
                                results["matched_entities"].append({
                                    "id": row["entity_id"],
                                    "name": row["entity_name"]
                                })
                                break # Match found for this row
            except Exception as e:
                print(f"[Lookup Engine Fallback] Failed to read entities CSV: {str(e)}")
                
        # 2. Metrics Lookup
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        for metric_kw in extracted_data.get("metrics", []):
                            kw_l = metric_kw.lower()
                            if kw_l in row["metric_name"].lower() or kw_l in row.get("synonyms", "").lower():
                                results["matched_metrics"].append({
                                    "id": row["metric_id"],
                                    "name": row["metric_name"],
                                    "calc_type": row.get("calculation_type", "sum")
                                })
                                break # Match found for this row
            except Exception as e:
                print(f"[Lookup Engine Fallback] Failed to read metrics CSV: {str(e)}")
                
        # 3. Relationships Lookup
        if len(results["matched_entities"]) > 1 and os.path.exists(relationships_path):
            try:
                entity_names = {e["name"].lower() for e in results["matched_entities"]}
                with open(relationships_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        src = row["source_entity"].lower()
                        tgt = row["target_entity"].lower()
                        if src in entity_names and tgt in entity_names:
                            results["relationships"].append({
                                "source": row["source_entity"],
                                "target": row["target_entity"],
                                "type": row.get("relationship_type", "join")
                            })
            except Exception as e:
                print(f"[Lookup Engine Fallback] Failed to read relationships CSV: {str(e)}")
                
        print("\n--- Metadata Lookup (Local Fallback) ---")
        print(f"Results: {results}")
        return results

