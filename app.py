"""Flask API for scoring one campaign idea."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import os
import time
import logging

from flask import Flask, jsonify, request

from functions import full_analysis, load_baseline_dict, load_local_causal_lm


BASE_DIR = Path(__file__).resolve().parent
BASELINE_PATH = BASE_DIR / "baseline_full.json"

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
baseline = load_baseline_dict(BASELINE_PATH)
# Initialize the local model during instance startup. The loader caches the
# tokenizer and model globally, so requests do not reload it.
model_started = time.perf_counter()
load_local_causal_lm()
app.logger.info("score_timing stage=model_startup seconds=%.3f", time.perf_counter() - model_started)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = os.environ.get("CORS_ALLOW_ORIGIN", "*")
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def _public_score_result(result: Dict[str, Any]) -> Dict[str, Any]:
    parts = result.get("parts", {})
    return {
        "idea": parts.get("idea"),
        "standardized_idea": parts.get("standardized_idea"),
        "components": parts.get("components_idea"),
        "scores": result.get("scores", {}),
    }


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/score_campaign", methods=["POST", "OPTIONS"], endpoint="score_campaign")
def score_campaign():
    if request.method == "OPTIONS":
        return "", 204

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be JSON: {\"idea\": \"...\"}"}), 400

    extra_keys = sorted(set(payload) - {"idea"})
    if extra_keys:
        return jsonify({"error": "Only the `idea` field is accepted.", "extra_fields": extra_keys}), 400

    idea = payload.get("idea")
    if not isinstance(idea, str) or not idea.strip():
        return jsonify({"error": "`idea` must be a non-empty string."}), 400

    try:
        started = time.perf_counter()
        result = full_analysis(idea.strip(), baseline)
        app.logger.info("score_timing stage=request_total seconds=%.3f", time.perf_counter() - started)
    except Exception as exc:
        app.logger.exception("Campaign scoring failed")
        return jsonify({"error": "Campaign scoring failed.", "detail": f"{type(exc).__name__}: {exc}"}), 500

    return jsonify(_public_score_result(result))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
