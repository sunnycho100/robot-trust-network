"""Readable activity lines and a detailed debug trace for the native window."""

import json
import traceback
from datetime import datetime
from pathlib import Path

TITLES = {
    "user": "YOU",
    "request": "REQUEST",
    "decision": "ROBOT DECIDES",
    "skill": "ROBOT",
    "result": "RESULT",
    "assistant": "ROBOT SAYS",
    "error": "ERROR",
    "controller_error": "ERROR",
    "system": "SYSTEM",
}


def exception_details(exc):
    """Local diagnostics without frame locals, credentials or model reasoning."""
    return {
        "exception_type": type(exc).__name__,
        "stack": [
            {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
            for frame in traceback.extract_tb(exc.__traceback__)
        ],
    }


def format_message(row, detailed=False):
    prefix = datetime.fromtimestamp(row["time"]).strftime("%H:%M:%S") if row.get("time") else ""
    role = row["role"]
    text = str(row.get("text", ""))
    details = row.get("details")
    if role == "result":
        ok = row.get("ok", isinstance(details, dict) and details.get("ok"))
        text = ("✓ " if ok else "✗ ") + text
        if not ok and isinstance(details, dict) and details.get("suggestion"):
            text += "\n  Suggestion: " + str(details["suggestion"])
    if role == "skill":
        text = "· " + text
    title = f"{prefix} {TITLES.get(role, role.upper())}".strip()
    if detailed and details:
        text += "\n" + json.dumps(details, indent=2, default=str)
    return title + "\n" + text
