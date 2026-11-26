"""Structured logging with a privacy rule enforced in code, not in habit.

Every request writes exactly one JSON line. The line contains counts, timings,
cost, status and a hashed caller id. It never contains the question text, the
answer text, document contents, email addresses or order notes. Tests assert
that, so the rule cannot silently regress.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Request

from app.config import settings

LOG_FILE = settings.log_dir / "app.jsonl"
REDACTED = "<redacted>"

# Belt and braces: if a message ever contains one of these keys, the value is
# replaced before it reaches disk.
SENSITIVE_KEYS = {
    "question",
    "text",
    "answer",
    "query",
    "email",
    "customer",
    "notes",
    "authorization",
    "api_key",
    "password",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            for key, value in extra.items():
                payload[key] = REDACTED if key.lower() in SENSITIVE_KEYS else value
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging() -> logging.Logger:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(JsonFormatter())
    logger.addHandler(stream)
    logger.propagate = False
    return logger


log = setup_logging()


def caller_id(request: Request) -> str:
    """A stable, non-reversible identifier for rate limiting and logs."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if settings.trust_proxy_headers and forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"
    digest = hashlib.sha256(f"{settings.log_salt}:{ip}".encode()).hexdigest()
    return digest[:16]


def log_event(event: str, **fields) -> None:
    log.info(event, extra={"fields": fields})


def iter_log_records(path: Path = LOG_FILE) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
