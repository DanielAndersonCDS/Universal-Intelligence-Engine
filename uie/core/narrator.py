"""Optional LLM narrator. It only ever sees finished fact statements (never raw data)
and its output is rejected if it contains any number the fact sheet does not support."""
from __future__ import annotations
from . import ollama, verifier
from .report import template_summary

SYSTEM = ("You rewrite verified analytical findings into a short executive summary (max 120 words). "
          "Use ONLY the statements provided. Do not add any number, name, date, or cause that is not in them. "
          "Copy numbers exactly as written. Say 'associated with', never 'caused by'. Plain prose, no lists.")


def summarize(sheet, model: str, host: str = ollama.DEFAULT_HOST, retries: int = 1):
    facts = sheet.ranked()[:12] or sheet.facts[:12]
    user = (f"Dataset: {sheet.dataset}, {sheet.rows} records, {sheet.cols} columns.\nFindings:\n"
            + "\n".join(f"[{f.id}] {f.narrate()}" for f in facts))
    try:
        for _ in range(retries + 1):
            text = ollama.chat(model, SYSTEM, user, host)
            bad = verifier.unsupported_numbers(text, sheet)
            if not bad:
                return text, f"llm:{model} (verified)"
        return template_summary(sheet), f"template (LLM output had unverified numbers: {bad[:3]})"
    except Exception as e:
        return template_summary(sheet), f"template (LLM unavailable: {str(e)[:80]})"
