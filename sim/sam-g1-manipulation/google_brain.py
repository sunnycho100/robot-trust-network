"""Direct Google model API adapter; credentials stay on the local server."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


class GoogleBrain:
    def __init__(self, model="gemini-3.8-flash", timeout=90):
        self.model = model
        self.timeout = timeout
        self.usage_path = Path("runs/agent/api_usage.jsonl")
        self.key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.key:
            raise RuntimeError(
                "Set GEMINI_API_KEY or GOOGLE_API_KEY in the launch environment"
            )
        if not all(c.isalnum() or c in "-._" for c in model):
            raise ValueError("Invalid model ID")

    @staticmethod
    def image_part(image: bytes, mime="image/jpeg"):
        return {
            "inlineData": {"mimeType": mime, "data": base64.b64encode(image).decode()}
        }

    def generate(self, contents, system, declarations, force_tool=False):
        usage_path = self.usage_path
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"maxOutputTokens": 4096},
            "tools": [{"functionDeclarations": declarations}],
            "toolConfig": {
                "functionCallingConfig": {"mode": "ANY" if force_tool else "AUTO"}
            },
        }
        if self.model.startswith("gemini-3"):
            payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "LOW"}
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            data=json.dumps(payload).encode(),
            headers={"x-goog-api-key": self.key, "Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        result = None
        error = None
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            # Only the response status/message is retained, never headers or key.
            detail = (
                exc.read()
                .decode(errors="replace")[:1000]
                .replace(self.key, "[redacted]")
            )
            error = f"HTTP {exc.code}"
            raise RuntimeError(f"Google API HTTP {exc.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            error = type(exc).__name__
            raise RuntimeError(
                f"Google API connection failed: {type(exc).__name__}"
            ) from None
        finally:
            usage_path.parent.mkdir(parents=True, exist_ok=True)
            usage = {
                "time": time.time(),
                "model": self.model,
                "latency_seconds": time.monotonic() - started,
                "usage": result.get("usageMetadata", {}) if result else {},
                "finish_reasons": [
                    c.get("finishReason") for c in result.get("candidates", [])
                ]
                if result
                else [],
                "error": error,
                "image_instances": sum(
                    "inlineData" in p for c in contents for p in c.get("parts", [])
                ),
            }
            with usage_path.open("a") as stream:
                stream.write(json.dumps(usage) + "\n")
        candidates = result.get("candidates", [])
        if not candidates or not candidates[0].get("content", {}).get("parts"):
            raise RuntimeError(
                "Google API returned no usable content; task was not executed"
            )
        if candidates[0].get("finishReason") not in (None, "STOP"):
            raise RuntimeError(
                f"Incomplete Google response: {candidates[0].get('finishReason')}"
            )
        # Preserve the entire model content for the next API turn, including
        # opaque thought signatures. Never expose hidden thoughts to the UI.
        return result


def public_parts(result):
    parts = result["candidates"][0]["content"]["parts"]
    text = "\n".join(p["text"] for p in parts if "text" in p and not p.get("thought"))
    calls = [p["functionCall"] for p in parts if "functionCall" in p]
    return text, calls


def probe(path, output, model="gemini-3.8-flash"):
    brain = GoogleBrain(model)
    declaration = {
        "name": "report_scene",
        "description": "Report what is actually visible in the image.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "robot_visible": {"type": "BOOLEAN"},
                "object_color": {"type": "STRING"},
                "description": {"type": "STRING"},
            },
            "required": ["robot_visible", "object_color", "description"],
        },
    }
    contents = [
        {
            "role": "user",
            "parts": [
                {
                    "text": "Inspect this simulation image and call report_scene with what you see."
                },
                brain.image_part(path.read_bytes(), "image/png"),
            ],
        }
    ]
    system = "You inspect a robotics simulation image. Use the supplied tool to report observations."
    result = brain.generate(contents, system, [declaration], force_tool=True)
    _, calls = public_parts(result)
    if len(calls) != 1 or calls[0]["name"] != "report_scene":
        raise RuntimeError(
            "Image/tool capability probe did not return the expected function call"
        )
    contents.append(result["candidates"][0]["content"])
    response = {"name": "report_scene", "response": {"accepted": True}}
    if "id" in calls[0]:
        response["id"] = calls[0]["id"]
    contents.append({"role": "user", "parts": [{"functionResponse": response}]})
    followup = brain.generate(
        contents,
        system + " The observation was accepted. Reply briefly in text.",
        [declaration],
    )
    report = {
        "requested_model": brain.model,
        "model_version": result.get("modelVersion"),
        "image_and_function_call": calls[0],
        "function_response_roundtrip": public_parts(followup)[0],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument(
        "--output", type=Path, default=Path("runs/flash-capabilities.json")
    )
    args = parser.parse_args()
    probe(args.image, args.output, args.model)
