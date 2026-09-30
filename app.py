"""Flask API for scoring one campaign idea."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import os
import time
import logging
import random
import re

from flask import Flask, jsonify, request
from google.genai import types

from functions import (
    full_analysis,
    full_analysis_batch_fast,
    batch_score_surprise,
    GEMINI_MODEL,
    get_gemini_client,
    gemini_embeddings_batch,
    gemini_json,
    generate_surprise_thoughtstarters,
    load_baseline_dict,
    load_local_causal_lm,
    score_solution_surprise,
)


BASE_DIR = Path(__file__).resolve().parent
BASELINE_PATH = BASE_DIR / "baseline_full.json"

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
baseline = load_baseline_dict(BASELINE_PATH)
SURPRISE_CANDIDATE_COUNT = 50
TENSION_CANDIDATE_COUNT = 20
SURPRISE_BATCH_SIZE = 32
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


def _fast_options(payload: Dict[str, Any]) -> tuple[int, bool, str | None]:
    """Read fast-route controls: no paraphrases and no web search by default."""
    paraphrase_count = payload.get("paraphrase_count", 0)
    use_web_search = payload.get("use_web_search", False)
    if isinstance(paraphrase_count, bool) or not isinstance(paraphrase_count, int) or paraphrase_count < 0:
        return 0, False, "`paraphrase_count` must be a non-negative integer."
    if not isinstance(use_web_search, bool):
        return 0, False, "`use_web_search` must be a boolean."
    return paraphrase_count, use_web_search, None


def _campaign_payload(payload: Any) -> tuple[str | None, str | None]:
    """Validate the compact campaign-only request used by the 8Ball routes."""
    if not isinstance(payload, dict):
        return None, 'Request body must be JSON: {"campaign": "..."}'
    campaign = payload.get("campaign")
    if not isinstance(campaign, str) or not campaign.strip():
        return None, "`campaign` must be a non-empty string."
    return campaign.strip(), None


def _clean_generated_line(value: object) -> str:
    if isinstance(value, dict) and isinstance(value.get("idea"), str):
        value = value["idea"]
    line = str(value).strip()
    return re.sub(r"^\s*(?:[-*]\s*|\d+[.)]\s*)", "", line).strip("`\" ")


def _unique_strings(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    result = []
    seen = set()
    for value in values:
        cleaned = _clean_generated_line(value)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            result.append(cleaned)
            seen.add(key)
    return result


def _generate_8ball_terms(campaign: str, count: int = SURPRISE_CANDIDATE_COUNT) -> list[str]:
    system = (
        "Return exactly the requested number of unique English words, one word per line. "
        "Do not use JSON, bullets, numbering, commas, explanations, or multi-word phrases. Choose "
        "concrete, campaign-relevant associations: "
        "physical objects, materials, foods, containers, places, people, animals, tools, visible "
        "actions, or everyday behaviors. Think in distinct topic buckets and prefer one useful "
        "umbrella word for each bucket. Avoid narrow sibling clusters, abstract concepts, emotions, "
        "virtues, values, strategy terms, adjectives, and vague marketing language."
    )
    prompt = (
        f"Campaign idea:\n{campaign}\n\n"
        f"Generate exactly {count} different one-word answers associated with this campaign, one per line."
    )
    response = get_gemini_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=1.0,
            max_output_tokens=max(4096, count * 20),
        ),
    )
    answers = [
        answer
        for answer in _unique_strings((response.text or "").splitlines())
        if re.fullmatch(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", answer)
    ]
    if len(answers) < 15:
        raise ValueError(f"Gemini returned only {len(answers)} usable surprise terms.")
    return answers


def _generate_8ball_tension_ideas(campaign: str, count: int = TENSION_CANDIDATE_COUNT) -> list[str]:
    system = (
        "Return only valid JSON with key `ideas`, whose value is an array of distinct campaign "
        "execution ideas. Return exactly the requested number. Each idea must be a concrete action, "
        "object, place, service, interface, ritual, or media format that a real campaign could use. "
        "Explore semantic displacement by combining the campaign with unexpected domains such as "
        "buildings, public infrastructure, transport, weather, games, retail, archives, or household "
        "objects. Make each idea specific, use a distinct mechanism or format, keep a plausible bridge "
        "to the campaign, and avoid explanations or repeated mechanisms."
    )
    prompt = (
        f"Original campaign:\n{campaign}\n\n"
        f"Generate exactly {count} different displaced campaign executions."
    )
    result = gemini_json(prompt, system, temperature=1.1, max_output_tokens=max(4096, count * 35))
    ideas = _unique_strings(result.get("ideas"))
    if len(ideas) < 15:
        raise ValueError(f"Gemini returned only {len(ideas)} usable tension ideas.")
    return ideas


def _generate_8ball_jux_terms(campaign: str, count: int = 20) -> tuple[list[str], list[str]]:
    system = (
        "Return only valid JSON with keys `related` and `left_field`. Each value must be an array "
        f"of exactly {count} unique one-word or two-word associations. `related` must contain "
        "concrete, strong associations with the campaign. `left_field` must contain concrete words "
        "or short phrases with no meaningful association to the campaign, chosen from completely "
        "unexpected domains. Use nouns, objects, places, activities, materials, or visible things. "
        "Do not use abstract concepts, explanations, duplicates, or multi-word phrases longer than "
        "two words."
    )
    prompt = (
        f"Campaign idea:\n{campaign}\n\n"
        f"Generate exactly {count} related associations and exactly {count} completely left-field "
        "associations."
    )
    result = gemini_json(prompt, system, temperature=1.1, max_output_tokens=4096)
    related = _unique_strings(result.get("related"))
    left_field = _unique_strings(result.get("left_field"))
    if len(related) < count or len(left_field) < count:
        raise ValueError(
            f"Gemini returned {len(related)} related and {len(left_field)} left-field terms; "
            f"expected {count} of each."
        )
    return related[:count], left_field[:count]


def _jux_distances(related: list[str], left_field: list[str]) -> list[float]:
    terms = related + left_field
    vectors = []
    for start in range(0, len(terms), 100):
        vectors.extend(gemini_embeddings_batch(terms[start:start + 100]))

    distances = []
    for related_index in range(len(related)):
        related_vector = vectors[related_index]
        related_norm = sum(value * value for value in related_vector) ** 0.5
        for left_field_index in range(len(left_field)):
            left_field_vector = vectors[len(related) + left_field_index]
            left_field_norm = sum(value * value for value in left_field_vector) ** 0.5
            similarity = sum(
                a * b for a, b in zip(related_vector, left_field_vector)
            ) / (related_norm * left_field_norm)
            distances.append(1.0 - similarity)
    return distances


def _generate_underlying_observations(
    campaign: str, ideas: list[str]
) -> list[tuple[str, str]]:
    system = (
        "Return only valid JSON with keys `observations` and `formats`. Both values must be arrays "
        "with exactly one item for each supplied idea, in the same order. Each observation must be "
        "one concise sentence describing an existing human behavior, habit, situation, norm, or "
        "cultural truth that the idea could be built from. Each format must contain only the "
        "physical or experiential format, setting, or channel of the idea. Preserve the specific "
        "domain and key noun that make the format recognizable, such as the type of venue, object, "
        "device, platform, or public space. Exclude only the mechanism, visual treatment, message, "
        "copy, and campaign claim from the format. Do not describe the campaign execution in the observation, "
        "do not explain the strategy, and do not invent brand claims."
    )
    idea_lines = "\n".join(f"{index + 1}. {idea}" for index, idea in enumerate(ideas))
    prompt = (
        f"Original campaign:\n{campaign}\n\n"
        f"Top displaced ideas:\n{idea_lines}\n\n"
        "Return one underlying cultural observation for each idea in the same order."
    )
    result = gemini_json(prompt, system, temperature=0.3, max_output_tokens=2048)
    observations = [str(value).strip() for value in result.get("observations", [])]
    formats = [str(value).strip() for value in result.get("formats", [])]
    if (
        len(observations) != len(ideas)
        or len(formats) != len(ideas)
        or any(not observation for observation in observations)
        or any(not format_name for format_name in formats)
    ):
        raise ValueError(
            f"Gemini returned {len(observations)} observations and {len(formats)} formats; "
            f"expected {len(ideas)} of each."
        )
    return list(zip(observations, formats))


def _minmax(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if high == low:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def _displacements(campaign: str, candidates: list[str]) -> list[float]:
    embedding_inputs = [campaign] + candidates
    vectors = []
    for start in range(0, len(embedding_inputs), 100):
        vectors.extend(gemini_embeddings_batch(embedding_inputs[start:start + 100]))
    original = vectors[0]
    original_norm = sum(value * value for value in original) ** 0.5
    distances = []
    for vector in vectors[1:]:
        norm = sum(value * value for value in vector) ** 0.5
        similarity = sum(a * b for a, b in zip(original, vector)) / (original_norm * norm)
        distances.append(1.0 - similarity)
    return distances


def _top_bottom(ideas: list[str], scores: list[float]) -> dict[str, list[dict[str, object]]]:
    ranked = sorted(
        ({"idea": idea, "score": round(float(score), 4)} for idea, score in zip(ideas, scores)),
        key=lambda row: row["score"],
        reverse=True,
    )
    return {"top": ranked[:10], "bottom": list(reversed(ranked[-10:]))}


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


@app.route("/score_campaign_fast", methods=["POST", "OPTIONS"], endpoint="score_campaign_fast")
def score_campaign_fast():
    """Score with the same methodology while parallelizing independent stages."""
    if request.method == "OPTIONS":
        return "", 204

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be JSON: {\"idea\": \"...\"}"}), 400

    extra_keys = sorted(set(payload) - {"idea", "paraphrase_count", "use_web_search"})
    if extra_keys:
        return jsonify({"error": "Only `idea`, `paraphrase_count`, and `use_web_search` are accepted.", "extra_fields": extra_keys}), 400

    idea = payload.get("idea")
    if not isinstance(idea, str) or not idea.strip():
        return jsonify({"error": "`idea` must be a non-empty string."}), 400

    paraphrase_count, use_web_search, options_error = _fast_options(payload)
    if options_error:
        return jsonify({"error": options_error}), 400

    try:
        started = time.perf_counter()
        result = full_analysis(
            idea.strip(),
            baseline,
            fast=True,
            paraphrase_count=paraphrase_count,
            use_web_search=use_web_search,
        )
        app.logger.info("score_timing stage=fast_request_total seconds=%.3f", time.perf_counter() - started)
    except Exception as exc:
        app.logger.exception("Fast campaign scoring failed")
        return jsonify({"error": "Fast campaign scoring failed.", "detail": f"{type(exc).__name__}: {exc}"}), 500

    return jsonify(_public_score_result(result))


@app.route("/score_campaigns_fast", methods=["POST", "OPTIONS"], endpoint="score_campaigns_fast")
def score_campaigns_fast():
    """Score multiple ideas with batched local-model inference."""
    if request.method == "OPTIONS":
        return "", 204

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("ideas"), list):
        return jsonify({"error": "Request body must be JSON: {\"ideas\": [\"...\"]}"}), 400
    ideas = payload["ideas"]
    if not ideas or len(ideas) > 100 or any(not isinstance(idea, str) or not idea.strip() for idea in ideas):
        return jsonify({"error": "`ideas` must contain 1–100 non-empty strings."}), 400

    extra_keys = sorted(set(payload) - {"ideas", "paraphrase_count", "use_web_search"})
    if extra_keys:
        return jsonify({"error": "Only `ideas`, `paraphrase_count`, and `use_web_search` are accepted.", "extra_fields": extra_keys}), 400
    paraphrase_count, use_web_search, options_error = _fast_options(payload)
    if options_error:
        return jsonify({"error": options_error}), 400

    try:
        started = time.perf_counter()
        results = full_analysis_batch_fast(
            [idea.strip() for idea in ideas],
            baseline,
            paraphrase_count=paraphrase_count,
            use_web_search=use_web_search,
        )
        app.logger.info("score_timing stage=batch_fast_request_total ideas=%d seconds=%.3f", len(ideas), time.perf_counter() - started)
    except Exception as exc:
        app.logger.exception("Fast batch campaign scoring failed")
        return jsonify({"error": "Fast batch campaign scoring failed.", "detail": f"{type(exc).__name__}: {exc}"}), 500

    return jsonify({"results": [_public_score_result(result) for result in results]})


@app.route("/8ball_surprise", methods=["POST", "OPTIONS"])
def eightball_surprise():
    """Return the top 10 campaign associations by combined score."""
    if request.method == "OPTIONS":
        return "", 204

    campaign, validation_error = _campaign_payload(request.get_json(silent=True))
    if validation_error:
        return jsonify({"error": validation_error}), 400

    try:
        terms = _generate_8ball_terms(campaign)
        context = f"Campaign idea: {campaign}\n\nWe want to do something with"
        surprise_results = batch_score_surprise(
            [(context, term) for term in terms],
            batch_size=SURPRISE_BATCH_SIZE,
        )
        surprise_scores = [float(result["average_surprise_bits"]) for result in surprise_results]
        displacement_scores = _displacements(campaign, terms)
        surprise_normalized = _minmax(surprise_scores)
        displacement_normalized = _minmax(displacement_scores)
        combined_scores = [
            (surprise_score + displacement_score) / 2
            for surprise_score, displacement_score in zip(
                surprise_normalized,
                displacement_normalized,
            )
        ]
        ranked = _top_bottom(terms, combined_scores)
        return jsonify({"top": ranked["top"][:10]})
    except Exception as exc:
        app.logger.exception("8Ball surprise scoring failed")
        return jsonify({"error": "8Ball surprise scoring failed.", "detail": f"{type(exc).__name__}: {exc}"}), 500


@app.route("/8ball_JUX", methods=["POST", "OPTIONS"])
def eightball_jux():
    """Return the 10 most semantically distant related/left-field pairings."""
    if request.method == "OPTIONS":
        return "", 204

    campaign, validation_error = _campaign_payload(request.get_json(silent=True))
    if validation_error:
        return jsonify({"error": validation_error}), 400

    try:
        related, left_field = _generate_8ball_jux_terms(campaign)
        distances = _jux_distances(related, left_field)
        combinations = [
            f"{related_term} + {left_field_term}"
            for related_term in related
            for left_field_term in left_field
        ]
        normalized = _minmax(distances)
        ranked = _top_bottom(combinations, normalized)
        return jsonify({"top": ranked["top"][:10]})
    except Exception as exc:
        app.logger.exception("8Ball JUX scoring failed")
        return jsonify({"error": "8Ball JUX scoring failed.", "detail": f"{type(exc).__name__}: {exc}"}), 500


@app.route("/8ball_tension", methods=["POST", "OPTIONS"])
def eightball_tension():
    """Return the top 10 campaign executions by normalized tension."""
    if request.method == "OPTIONS":
        return "", 204

    campaign, validation_error = _campaign_payload(request.get_json(silent=True))
    if validation_error:
        return jsonify({"error": validation_error}), 400

    try:
        ideas = _generate_8ball_tension_ideas(campaign)
        displacement_scores = _displacements(campaign, ideas)
        normalized = _minmax(displacement_scores)
        tension_scores = [
            max(0.1, min(0.9, 0.1 + 0.8 * score + random.uniform(-0.05, 0.05)))
            for score in normalized
        ]
        ranked = _top_bottom(ideas, tension_scores)
        top_ideas = [row["idea"] for row in ranked["top"][:10]]
        observations_and_formats = _generate_underlying_observations(campaign, top_ideas)
        for row, original_idea, (observation, format_name) in zip(
            ranked["top"][:10], top_ideas, observations_and_formats
        ):
            format_name = format_name.rstrip(" .!?;:")
            format_text = format_name[0].lower() + format_name[1:]
            row["idea"] = f"{observation} How would you use a {format_text}?"
            row["idea_underlying"] = original_idea
        return jsonify({"top": ranked["top"][:10]})
    except Exception as exc:
        app.logger.exception("8Ball tension scoring failed")
        return jsonify({"error": "8Ball tension scoring failed.", "detail": f"{type(exc).__name__}: {exc}"}), 500


@app.route("/suprise_thoughtstarters", methods=["POST", "OPTIONS"])
def suprise_thoughtstarters():
    """Generate more/less chaotic solutions and score their surprisal."""
    if request.method == "OPTIONS":
        return "", 204

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({
            "error": (
                "Request body must be JSON: "
                '{"cultural_observation": "...", "brand_problem": "..."}'
            )
        }), 400

    extra_keys = sorted(set(payload) - {"cultural_observation", "brand_problem", "problem_to_solve"})
    if extra_keys:
        return jsonify({"error": "Only cultural observation and brand problem fields are accepted.", "extra_fields": extra_keys}), 400

    cultural_observation = payload.get("cultural_observation")
    brand_problem = payload.get("brand_problem", payload.get("problem_to_solve"))
    if not isinstance(cultural_observation, str) or not cultural_observation.strip():
        return jsonify({"error": "`cultural_observation` must be a non-empty string."}), 400
    if not isinstance(brand_problem, str) or not brand_problem.strip():
        return jsonify({"error": "`brand_problem` must be a non-empty string."}), 400

    try:
        started = time.perf_counter()
        generated = generate_surprise_thoughtstarters(
            cultural_observation.strip(),
            brand_problem.strip(),
        )
        response = {
            group: [
                {
                    "solution": solution,
                    "surprise_score": score_solution_surprise(solution)["mean"],
                }
                for solution in solutions
            ]
            for group, solutions in generated.items()
        }
        app.logger.info("score_timing stage=thoughtstarters_total seconds=%.3f", time.perf_counter() - started)
    except Exception as exc:
        app.logger.exception("Surprise thoughtstarter generation failed")
        return jsonify({"error": "Surprise thoughtstarter generation failed.", "detail": f"{type(exc).__name__}: {exc}"}), 500

    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
