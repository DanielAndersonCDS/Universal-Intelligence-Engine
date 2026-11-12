"""Rejects LLM text containing any number that is not backed by the fact sheet."""
from __future__ import annotations
import re

NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _parse(text: str):
    text = re.sub(r"\bF\d{3}\b", "", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s", "", text)
    for m in NUM.finditer(text):
        tok = m.group(0).rstrip(".,")
        try:
            v = float(tok.replace(",", ""))
        except ValueError:
            continue
        dec = len(tok.split(".")[1]) if "." in tok else 0
        yield v, dec


def _walk(o, out):
    if isinstance(o, bool) or o is None:
        return
    if isinstance(o, (int, float)):
        out.extend([float(o), float(o) * 100])
    elif isinstance(o, str):
        out.extend(v for v, _ in _parse(o))
    elif isinstance(o, dict):
        for v in o.values():
            _walk(v, out)
    elif isinstance(o, (list, tuple)):
        for v in o:
            _walk(v, out)


def allowed_numbers(sheet):
    out = [float(sheet.rows), float(sheet.cols), float(len(sheet.facts)), float(sheet.patterns()), float(sheet.anomalies())]
    for f in sheet.facts:
        _walk(f.metrics, out)
        out.extend(v for v, _ in _parse(f.narrate()))
    return out


def unsupported_numbers(text: str, sheet):
    allowed = allowed_numbers(sheet)
    bad = []
    for v, dec in _parse(text):
        tol = 0.5 * 10 ** (-dec) + 1e-9
        if not any(abs(v - a) <= tol for a in allowed):
            bad.append(v)
    return bad
