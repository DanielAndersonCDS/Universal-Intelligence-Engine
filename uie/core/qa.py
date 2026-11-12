"""Ask the data. The LLM (optional) only sees the schema and writes SQL; results never go back to it."""
from __future__ import annotations
import re
from . import ollama

START = re.compile(r"^\s*(select|with|describe|summarize|from|show)\b", re.I)
FORBID = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|export|import|install|load|"
                    r"pragma|call|truncate|grant|set)\b|\bread_\w+|\bglob\b|\w+_scan\b|\b(from|join)\s+['\"]", re.I)
SHORTCUTS = {"head": "SELECT * FROM data LIMIT 20", "schema": "DESCRIBE data", "count": "SELECT count(*) AS rows FROM data"}


def validate_sql(sql: str) -> str:
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        raise ValueError("Only a single statement is allowed.")
    if not START.match(s):
        raise ValueError("Only read-only SELECT / WITH / DESCRIBE / SUMMARIZE queries are allowed.")
    if FORBID.search(s):
        raise ValueError("Query contains a disallowed keyword (read-only mode).")
    return s


def run_sql(con, sql: str, limit: int = 1000):
    cur = con.cursor()
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchmany(limit)
    return cols, rows, bool(cur.fetchmany(1))


def _clean(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip()


def answer(engine, question: str, use_llm=False, model="phi4-mini", host=ollama.DEFAULT_HOST) -> dict:
    q = question.strip()
    via = "sql"
    if q.lower() in SHORTCUTS:
        sql = SHORTCUTS[q.lower()]
    elif START.match(q):
        sql = q
    elif use_llm:
        schema = ", ".join(f'"{n}" {t}' for n, t in engine.schema())
        sys_p = ("You translate questions into one DuckDB SQL SELECT statement. The only table is data("
                 + schema + "). Return only the SQL, no explanation.")
        sql = _clean(ollama.chat(model, sys_p, q, host))
        via = f"llm:{model}"
    else:
        raise ValueError("Type SQL directly (SELECT ...), use head / schema / count, or enable the local LLM to ask in plain English.")
    sql = validate_sql(sql)
    cols, rows, more = run_sql(engine.src.con, sql)
    return {"sql": sql, "cols": cols, "rows": rows, "truncated": more, "via": via}
