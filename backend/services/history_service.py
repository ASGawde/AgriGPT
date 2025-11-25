# backend/services/history_service.py

import json
import os
import threading
from pathlib import Path
from datetime import datetime

LOG_PATH = Path("backend/data/query_log.json")
TMP_PATH = Path("backend/data/query_log_tmp.json")

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Thread lock (same-process)
_log_lock = threading.Lock()

# Rotate logs when size > 5MB
MAX_LOG_SIZE_BYTES = 5 * 1024 * 1024  


def _sanitize_entry(entry: dict) -> dict:
    """
    Ensures all values are JSON-serializable UTF-8 strings.
    Prevents logging failures due to emojis or invalid surrogate pairs.
    """
    clean = {}
    for key, val in entry.items():
        if isinstance(val, (str, int, float, bool)) or val is None:
            clean[key] = val
        else:
            # Approximate complex values
            clean[key] = str(val)
    return clean


def _atomic_write(data: list):
    """
    Safely writes to a temporary file, then atomically replaces the log.
    Prevents log corruption from partial writes.
    """
    with open(TMP_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Atomic replace
    os.replace(TMP_PATH, LOG_PATH)


def log_interaction(entry: dict):
    """
    Append a single query/response record to query_log.json.
    Thread-safe, atomic, and corruption-proof.
    """

    clean_entry = _sanitize_entry(entry)
    if "timestamp" not in clean_entry:
        clean_entry["timestamp"] = datetime.utcnow().isoformat()

    with _log_lock:
        try:
            # Step 1 — Load existing logs (if valid JSON)
            if LOG_PATH.exists():
                with open(LOG_PATH, "r", encoding="utf-8") as f:
                    logs = json.load(f)
                if not isinstance(logs, list):
                    logs = []
            else:
                logs = []

        except Exception:
            # JSON corrupt → reset to empty
            logs = []

        # Step 2 — Append
        logs.append(clean_entry)

        # Step 3 — Rotate if too large
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > MAX_LOG_SIZE_BYTES:
            archive_path = LOG_PATH.with_suffix(".archive.json")
            os.replace(LOG_PATH, archive_path)
            logs = [clean_entry]   # start fresh after rotation

        # Step 4 — Atomic write
        try:
            _atomic_write(logs)
        except Exception as e:
            print(f"[LOG ERROR] Failed to write log: {e}")
