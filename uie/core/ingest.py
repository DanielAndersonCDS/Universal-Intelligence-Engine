from __future__ import annotations
import tempfile
from pathlib import Path
import duckdb


def _esc(s: str) -> str:
    return s.replace("'", "''")


class DataSource:
    """Wraps a DuckDB connection exposing the chosen dataset as a view (default name `raw`).
    Several DataSources can share one connection (reconciliation, drift, train/test checks)."""

    def __init__(self, con=None, view: str = "raw"):
        if con is None:
            con = duckdb.connect(":memory:")
            spill = Path(tempfile.gettempdir()) / "uie_spill"
            spill.mkdir(exist_ok=True)
            con.execute(f"SET temp_directory='{_esc(spill.as_posix())}'")
        self.con = con
        self.view = view
        self.alias = f"ext_{view}"
        self.path = None
        self.name = ""
        self.size_bytes = 0
        self.tables: list[str] = []
        self.table = None
        self.fallback_used = False
        self._src_sql = None
        self._csv_path = None
        self.files: list[str] = []

    def open(self, path, table=None):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        self.path, self.name = str(p), p.name
        ext = p.suffix.lower()
        P = _esc(p.as_posix())
        if p.is_dir():
            pq = list(p.rglob("*.parquet"))
            if pq:
                self.files = [str(f) for f in pq]
                self.size_bytes = sum(f.stat().st_size for f in pq)
                self._view(f"read_parquet('{P}/**/*.parquet')")
            else:
                cs = list(p.rglob("*.csv"))
                if not cs:
                    raise ValueError("Folder contains no .parquet or .csv files")
                self.files = [str(f) for f in cs]
                self.size_bytes = sum(f.stat().st_size for f in cs)
                self._csv_path = f"{P}/**/*.csv"
                self._view(f"read_csv_auto('{self._csv_path}', sample_size=100000)")
            return self
        self.files = [str(p)]
        self.size_bytes = p.stat().st_size
        if ext in (".csv", ".tsv", ".txt"):
            self._csv_path = P
            self._view(f"read_csv_auto('{P}', sample_size=100000)")
        elif ext == ".parquet":
            self._view(f"read_parquet('{P}')")
        elif ext in (".json", ".jsonl", ".ndjson"):
            self._view(f"read_json_auto('{P}')")
        elif ext == ".xlsx":
            self._view(f"read_xlsx('{P}')")
        elif ext in (".duckdb", ".db", ".sqlite", ".sqlite3"):
            if ext == ".duckdb":
                self.con.execute(f"ATTACH '{P}' AS {self.alias} (READ_ONLY)")
            else:
                self.con.execute("INSTALL sqlite; LOAD sqlite;")
                self.con.execute(f"ATTACH '{P}' AS {self.alias} (TYPE sqlite, READ_ONLY)")
            rows = self.con.execute(
                f"SELECT table_schema, table_name FROM information_schema.tables WHERE table_catalog='{self.alias}' "
                "AND table_schema NOT IN ('information_schema','pg_catalog') ORDER BY 2").fetchall()
            self.tables = [f"{s}.{t}" for s, t in rows]
            if not self.tables:
                raise ValueError("Database has no tables")
            self.use_table(table or self.tables[0])
        else:
            raise ValueError(f"Unsupported file type: {ext or p.name}")
        return self

    def use_table(self, label: str):
        schema, name = label.split(".", 1)
        self.table = label
        self._view(f'{self.alias}."{schema}"."{name}"')

    def _view(self, src_sql: str):
        self._src_sql = src_sql
        self.con.execute(f"CREATE OR REPLACE VIEW {self.view} AS SELECT * FROM {src_sql}")
        self.con.execute(f"SELECT * FROM {self.view} LIMIT 1").fetchall()

    def source_sql(self) -> str:
        return f"CREATE VIEW {self.view} AS SELECT * FROM {self._src_sql}"

    def fallback_text(self) -> bool:
        """Reload a CSV with every column as text when type detection fails mid-file."""
        if not self._csv_path or self.fallback_used:
            return False
        self.fallback_used = True
        self._view(f"read_csv('{self._csv_path}', all_varchar=true, sample_size=100000)")
        return True


def open_pair(left, right, table_left=None, table_right=None):
    """Open two datasets on one connection as views left_raw / right_raw. Returns (con, left, right)."""
    a = DataSource(view="left_raw").open(left, table_left)
    b = DataSource(con=a.con, view="right_raw").open(right, table_right)
    return a.con, a, b
