import sqlite3
import pandas as pd
import json


def get_rich_schema(db_path: str) -> str:
    """Return a rich description of all tables: columns, types, row counts, sample values, inferred FK relationships."""
    result = {}

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        if not tables:
            return json.dumps({"error": "Database is empty — no tables found."})

        all_columns: dict[str, list[str]] = {}

        for table in tables:
            cursor.execute(f"PRAGMA table_info([{table}]);")
            columns = cursor.fetchall()

            cursor.execute(f"SELECT COUNT(*) FROM [{table}];")
            row_count = cursor.fetchone()[0]

            table_info: dict = {"row_count": row_count, "columns": {}}

            for col in columns:
                col_name = col[1]
                col_type = col[2]

                cursor.execute(f"SELECT COUNT(*) FROM [{table}] WHERE [{col_name}] IS NULL;")
                null_count = cursor.fetchone()[0]

                cursor.execute(
                    f"SELECT DISTINCT [{col_name}] FROM [{table}] "
                    f"WHERE [{col_name}] IS NOT NULL LIMIT 10;"
                )
                sample_vals = [str(row[0]) for row in cursor.fetchall()]

                table_info["columns"][col_name] = {
                    "type": col_type,
                    "null_count": null_count,
                    "sample_values": sample_vals,
                }

            result[table] = table_info
            all_columns[table] = [col[1] for col in columns]

        # Infer FK relationships by matching column names across tables
        relationships = []
        table_list = list(all_columns.keys())
        for i, t1 in enumerate(table_list):
            for t2 in table_list[i + 1:]:
                shared = set(all_columns[t1]) & set(all_columns[t2])
                for col in shared:
                    relationships.append(
                        f"{t1}.{col} likely joins with {t2}.{col}"
                    )

        if relationships:
            result["_inferred_relationships"] = relationships

    return json.dumps(result, indent=2)


def execute_sql(db_path: str, query: str) -> str:
    """Execute a SQLite SQL query. Returns results as a markdown table, or the error message if it fails."""
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(query, conn)
        if df.empty:
            return "Query executed successfully. Result: 0 rows returned."
        # Cap output at 100 rows to avoid flooding the context
        preview = df.head(100)
        note = f"\n_(showing {len(preview)} of {len(df)} rows)_" if len(df) > 100 else ""
        return preview.to_markdown(index=False) + note
    except Exception as e:
        return f"SQL_ERROR: {e}"


def get_distinct_values(db_path: str, table: str, column: str) -> str:
    """Return all distinct values and their row counts for a column. Use this before writing WHERE clauses."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT [{column}], COUNT(*) as count FROM [{table}] "
                f"GROUP BY [{column}] ORDER BY count DESC;"
            )
            rows = cursor.fetchall()
        result = {str(row[0]): int(row[1]) for row in rows}
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {e}"
