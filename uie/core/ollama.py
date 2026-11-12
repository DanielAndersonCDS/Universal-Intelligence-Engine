from __future__ import annotations
import json, re, urllib.request

DEFAULT_HOST = "http://localhost:11434"


def list_models(host=DEFAULT_HOST):
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=3) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return []


def chat(model, system, user, host=DEFAULT_HOST, timeout=180) -> str:
    payload = {"model": model, "stream": False, "options": {"temperature": 0},
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(host.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = json.loads(r.read())["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
