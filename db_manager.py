import sqlite3
import pandas as pd
from datetime import datetime
import os
from typing import List, Optional


class DatabaseManager:
    def __init__(self, db_dir: str = "databases"):
        os.makedirs(db_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.db_path = os.path.join(db_dir, f"database_{timestamp}.db")

    def add_table(self, df: pd.DataFrame, table_name: str) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql(table_name, conn, if_exists="replace", index=False)
            return True
        except Exception as e:
            print(f"Error adding table '{table_name}': {e}")
            return False

    def get_tables(self) -> List[str]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                return [row[0] for row in cursor.fetchall()]
        except Exception:
            return []

    def execute_query(self, query: str) -> Optional[pd.DataFrame]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            print(f"Query error: {e}")
            return None

    def is_empty(self) -> bool:
        return len(self.get_tables()) == 0

    def clear(self) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                for table in tables:
                    cursor.execute(f"DROP TABLE IF EXISTS [{table}];")
                conn.commit()
            return True
        except Exception as e:
            print(f"Error clearing database: {e}")
            return False
