# backend/services/history_service.py

import json
import threading
from pathlib import Path
from datetime import datetime

# Thread lock to prevent simultaneous writes
_log_lock = threading.Lock()

LOG_PATH = Path("backend/data/query_log.json")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_interaction(entry: dict):
    """
    Append a single query/response record to query_log.json.
    Safe for concurrent requests.
    """

    with _log_lock:
        # Step 1 — Read existing logs safely
        try:
            if LOG_PATH.exists():
                with open(LOG_PATH, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            else:
                logs = []
        except Exception:
            # Corrupt file or invalid JSON
            logs = []

        # Step 2 — Append new entry with timestamp (if missing)
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.utcnow().isoformat()

        logs.append(entry)

        # Step 3 — Write back atomically
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
