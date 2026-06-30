import sqlite3
import csv
import json
import os
import time
from utils.logger import log_step


def _csv_to_sqlite(conn, table_name, file_path):
    """Load a CSV file into a SQLite table using pure stdlib csv module (no pandas)."""
    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"  Warning: {file_path} is empty.")
        return 0

    columns = rows[0].keys()
    col_defs = ", ".join(f'"{c}" TEXT' for c in columns)
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    conn.execute(f'CREATE TABLE "{table_name}" ({col_defs})')

    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f'INSERT INTO "{table_name}" VALUES ({placeholders})',
        [tuple(row[c] for c in columns) for row in rows]
    )
    return len(rows)


def setup_database(db_path, metadata_dir):
    """
    Load CSV and JSON metadata into SQLite.
    Uses pure stdlib (csv, sqlite3, json) — no pandas dependency.
    """
    start_time = time.time()
    log_step("Starting SQLite database creation & metadata import...")
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)

    # Load CSVs
    csv_files = {
        'semantic_entities': 'semantic_entities.csv',
        'semantic_metrics': 'semantic_metrics.csv',
        'semantic_relationships': 'semantic_relationships.csv',
        'semantic_source_mappings': 'semantic_source_mappings.csv',
        'semantic_metric_mappings': 'semantic_metric_mappings.csv'
    }

    for table_name, filename in csv_files.items():
        file_path = os.path.join(metadata_dir, filename)
        if os.path.exists(file_path):
            csv_start = time.time()
            count = _csv_to_sqlite(conn, table_name, file_path)
            log_step(f"Loaded {count} records into '{table_name}' from {filename}", csv_start)
        else:
            log_step(f"Warning: {file_path} not found.")

    # Load Workspace Config JSON
    config_path = os.path.join(metadata_dir, 'workspace_config.json')
    if os.path.exists(config_path):
        json_start = time.time()
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        workspaces = config_data.get('workspaces', [])
        if workspaces:
            columns = workspaces[0].keys()
            col_defs = ", ".join(f'"{c}" TEXT' for c in columns)
            conn.execute('DROP TABLE IF EXISTS "workspace_config"')
            conn.execute(f'CREATE TABLE "workspace_config" ({col_defs})')
            placeholders = ", ".join("?" for _ in columns)
            conn.executemany(
                f'INSERT INTO "workspace_config" VALUES ({placeholders})',
                [tuple(str(ws.get(c, '')) for c in columns) for ws in workspaces]
            )
            log_step(f"Loaded {len(workspaces)} workspaces into 'workspace_config'", json_start)
    else:
        log_step(f"Warning: {config_path} not found.")

    conn.commit()
    conn.close()
    
    log_step("Completed SQLite database creation & metadata import", start_time)


if __name__ == "__main__":
    setup_database("../../db/metadata.db", "../../metadata")
