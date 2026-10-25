"""Original deterministic implementation for Local Knowledge (plotxyz-local-knowledge-agent)."""
from __future__ import annotations

import ast
import csv
import io
import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict, deque


class LogicError(ValueError):
    """Raised when supplied input cannot be processed."""


def _text(payload, key, min_len=1, max_len=20000):
    value = payload.get(key)
    if not isinstance(value, str):
        raise LogicError(f"'{key}' must be a string")
    value = value.strip()
    if len(value) < min_len:
        raise LogicError(f"'{key}' is required")
    if len(value) > max_len:
        raise LogicError(f"'{key}' exceeds {max_len} characters")
    return value


def _num(payload, key, default=None, lo=None, hi=None):
    value = payload.get(key, default)
    if value is None or value == "":
        if default is None:
            raise LogicError(f"'{key}' is required")
        return default
    if isinstance(value, bool):
        raise LogicError(f"'{key}' must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise LogicError(f"'{key}' must be numeric")
    if lo is not None and number < lo:
        raise LogicError(f"'{key}' must be at least {lo}")
    if hi is not None and number > hi:
        raise LogicError(f"'{key}' must be at most {hi}")
    return number


def _json(payload, key, expected=None, max_len=200000):
    raw = payload.get(key)
    if isinstance(raw, (dict, list)):
        data = raw
    elif isinstance(raw, str):
        if len(raw) > max_len:
            raise LogicError(f"'{key}' exceeds {max_len} characters")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LogicError(f"'{key}' is not valid JSON: {exc.msg} at line {exc.lineno}")
    else:
        raise LogicError(f"'{key}' must be JSON")
    if expected is not None and not isinstance(data, expected):
        name = expected.__name__ if hasattr(expected, "__name__") else str(expected)
        raise LogicError(f"'{key}' must be a JSON {name}")
    return data


def _csv(payload, key, max_rows=5000):
    text = _text(payload, key, max_len=2000000)
    try:
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise LogicError(f"'{key}' must include a header row")
        columns = [str(c).strip() for c in reader.fieldnames if c and str(c).strip()]
        if not columns:
            raise LogicError(f"'{key}' must include named columns")
        rows = []
        for index, raw in enumerate(reader):
            if index >= max_rows:
                raise LogicError(f"'{key}' exceeds {max_rows} rows")
            rows.append({col: (raw.get(col) or "").strip() for col in columns})
    except csv.Error as exc:
        raise LogicError(f"'{key}' is not valid CSV: {exc}")
    if not rows:
        raise LogicError(f"'{key}' must include at least one data row")
    return columns, rows


STOPWORDS = {'will', 'they', 'was', 'your', 'than', 'that', 'on', 'be', 'then', 'do', 'our', 'with', 'he', 'is', 'about', 'had', 'were', 'as', 'may', 'we', 'their', 'are', 'did', 'at', 'she', 'a', 'by', 'the', 'if', 'might', 'it', 'into', 'have', 'from', 'should', 'not', 'over', 'can', 'for', 'and', 'an', 'you', 'does', 'to', 'this', 'could', 'would', 'under', 'of', 'no', 'its', 'in', 'i', 'or', 'has'}


def _tokens(text):
    return [w for w in re.findall(r"[a-z0-9][a-z0-9_+-]{1,}", text.lower()) if w not in STOPWORDS]


def _sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -•\t") for p in parts if len(p.strip()) > 2]


def provider_status():
    """Report which optional model providers are configured (never exposes secrets)."""
    ollama = bool(os.getenv("OLLAMA_BASE_URL"))
    openai = bool(os.getenv("OPENAI_BASE_URL") and os.getenv("OPENAI_MODEL"))
    return {"ollama_compatible": ollama, "openai_compatible": openai}


def _llm(messages, temperature=0.2):
    """Call an explicitly configured OpenAI-compatible endpoint; return None when unavailable."""
    base = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "")
    if not base or not model:
        ollama = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
        if not ollama:
            return None
        base = f"{ollama}/v1"
        model = os.getenv("OLLAMA_MODEL", "")
    if not model:
        return None
    try:
        import httpx

        headers = {"Content-Type": "application/json"}
        key = os.getenv("OPENAI_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        with httpx.Client(timeout=20) as client:
            response = client.post(
                f"{base}/chat/completions",
                headers=headers,
                json={"model": model, "messages": messages, "temperature": temperature},
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"]).strip()
    except Exception:
        return None


def _result(summary, result, warnings=None):
    return {"summary": summary, "result": result, "warnings": warnings or []}


def agent_local_knowledge(payload):
    documents = _text(payload, "documents", max_len=120000)
    query = _text(payload, "query", max_len=1000)
    top_k = int(_num(payload, "top_k", default=3, lo=1, hi=5))
    sections = []
    if "###" in documents:
        for index, block in enumerate(re.split(r"\n?\s*###\s*", documents)):
            block = block.strip()
            if block:
                lines = block.splitlines()
                title = lines[0].strip("# ").strip() or f"Document {index}"
                sections.append((title, "\n".join(lines[1:]) if len(lines) > 1 else block))
    else:
        for index, block in enumerate([b for b in re.split(r"\n\s*\n", documents) if b.strip()], 1):
            sections.append((f"Document {index}", block.strip()))
    if not sections:
        raise LogicError("no readable documents were supplied")
    query_tokens = set(_tokens(query))
    passages = []
    for title, body in sections:
        words = body.split()
        for start in range(0, len(words), 60):
            chunk = " ".join(words[start:start + 60]).strip()
            if chunk:
                passages.append({"title": title, "text": chunk})
    scored = []
    for passage in passages:
        tokens = set(_tokens(passage["text"]))
        overlap = len(query_tokens & tokens) / max(1, len(query_tokens))
        phrase = 0.5 if query.lower() in passage["text"].lower() else 0
        scored.append((round(overlap + phrase, 3), passage))
    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:top_k]
    lines = ["**Extractive answer from approved documents**", ""]
    for rank, (score, passage) in enumerate(top, 1):
        excerpt = passage["text"][:420]
        lines.append(f"{rank}. **{passage['title']}** (relevance {score:.2f})")
        lines.append(f"   > {excerpt}")
    best = top[0][0] if top else 0
    findings = []
    if best <= 0:
        findings.append({"title": "No lexical match", "severity": "medium", "detail": "No passage shares keywords with the query. Try different terms or add documents."})
    return _result(
        f"Retrieved {len(top)} passage(s); best relevance {best:.2f}. Answers are extractive with source titles.",
        {
            "metrics": {"passages_indexed": len(passages), "retrieved": len(top), "best_score": best},
            "markdown": "\n".join(lines),
            "table": {"columns": ["Source", "Relevance", "Excerpt"], "rows": [[p["title"], s, p["text"][:120]] for s, p in top]},
            "findings": findings,
        },
        ["Keyword retrieval only; no embeddings or external search are used."],
    )


run = agent_local_knowledge
