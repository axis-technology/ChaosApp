"""Reusable functions for creative idea chaos scoring.

This module consolidates the core notebook functions:

- Step 2: semantic displacement from a baseline
- Step 3: contextual surprise with Qwen/Qwen3-4B-Base
- Step 4: relationship-based internal incongruity
- Gemini idea paraphrasing
- Gemini campaign component extraction

No API keys are stored here. Set one of these before using Gemini functions:

    os.environ["GOOGLE_CLOUD_PROJECT"] = "your-project-id"      # Vertex route
    os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

or:

    os.environ["GOOGLE_CLOUD_API_KEY"] = "your-api-key"         # Gemini API-key route
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import subprocess
import html
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, cast

import pandas as pd
import numpy as np
from google import genai
from google.genai import types


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Text and embedding models copied from NewProcess/exp2_idea_paraphrase_scores.ipynb.
GEMINI_MODEL = "gemini-3.7-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
#CAUSAL_LM_MODEL = "Qwen/Qwen3-4B-Base"
CAUSAL_LM_MODEL = os.environ.get("CAUSAL_LM_MODEL", "Qwen/Qwen3-0.6B-Base")

DEVICE = os.environ.get("CHAOS_ENGINE_DEVICE", "auto")
SURPRISE_BATCH_SIZE = 4
USE_AVERAGE_SURPRISE = True
NORMALIZATION_EPSILON = 1e-12

SAFETY_SETTINGS_OFF = [
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
]

_local_tokenizer = None
_local_model = None
_local_model_name = None
_embedding_cache: Dict[str, List[float]] = {}
_surprise_cache: Dict[Tuple[str, str, str], Dict[str, object]] = {}
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)


def _load_torch():
    import torch
    import torch.nn.functional as F

    return torch, F


def _resolve_device(torch_module) -> str:
    if DEVICE != "auto":
        return DEVICE
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def clean_text(text: object) -> str:
    """Collapse whitespace and coerce missing values into empty strings."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return " ".join(str(text).strip().split())


def count_words(text: str) -> int:
    """Count words with punctuation ignored."""
    return len(re.findall(r"\b[\w%'-]+\b", clean_text(text)))


def get_gcloud_project() -> Optional[str]:
    """Read the active gcloud project if one is configured."""
    try:
        project = subprocess.check_output(
            ["gcloud", "config", "get-value", "project"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
    return project if project and project != "(unset)" else None


def get_gemini_client(
    project: Optional[str] = None,
    location: Optional[str] = None,
    api_key: Optional[str] = None,
    require_api_key: bool = False,
) -> genai.Client:
    """Create a Gemini client.

    This follows the Exp 2 notebook behavior: use Vertex when a project is
    available, otherwise use an API key. `gemini-2.5-flash` and
    `gemini-embedding-001` work through the tested Vertex route.
    """
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT") or get_gcloud_project()
    location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    api_key = api_key or os.environ.get("GOOGLE_CLOUD_API_KEY")

    if api_key:
        return genai.Client(vertexai=False, api_key=api_key)
    if require_api_key:
        raise ValueError(
            "gemini-3.7-flash must use the Gemini API-key route in this project. "
            "Set GOOGLE_CLOUD_API_KEY before calling this function."
        )
    if project:
        return genai.Client(vertexai=True, project=project, location=location)
    raise ValueError("Set GOOGLE_CLOUD_PROJECT for Vertex or GOOGLE_CLOUD_API_KEY for Gemini API.")


def collect_gemini_stream(client: genai.Client, model: str, contents, config) -> str:
    """Collect streamed Gemini text into one string."""
    chunks = []
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
        if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
            continue
        chunks.append(chunk.text or "")
    return "".join(chunks)


def parse_json_object(text: str) -> Dict[str, object]:
    """Parse a JSON object, accepting fenced JSON from model output."""
    text = clean_text(text)
    if text.startswith("```json"):
        text = text.removeprefix("```json").removesuffix("```").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def gemini_json(
    prompt: str,
    system_instruction: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    use_web_search: bool = False,
    max_output_tokens: int = 8192,
) -> Dict[str, object]:
    """Call Gemini and parse a JSON response."""
    model = model or GEMINI_MODEL
    client = get_gemini_client(require_api_key=(model == "gemini-3.7-flash"))
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    config_kwargs = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "safety_settings": SAFETY_SETTINGS_OFF,
        "system_instruction": [types.Part.from_text(text=system_instruction)],
    }
    if use_web_search:
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

    config = types.GenerateContentConfig(**config_kwargs)
    response_text = collect_gemini_stream(client, model, contents, config)
    return parse_json_object(response_text)


def standardize_idea(raw_idea: str) -> str:
    system_instruction=  """
        You standardize marketing campaign ideas into simple, literal descriptions for consistent analysis.

    Rewrite the provided campaign idea as exactly ONE concise sentence.

    Required format:
    "A [1–2 word company category] brand [concrete tactic/action] to [purpose/outcome]."

    OR

    "A [1–2 word company category] brand [concrete tactic/action] with [mechanism/object]."

    Examples of company categories:

    * Nike → "shoe"
    * Coca-Cola → "beverage"
    * Spotify → "music streaming"
    * IKEA → "furniture"
    * Dove → "personal care"

    Rules:

    1. Always begin with: "A [company category] brand"
    2. The company category must be 1–2 words describing what the company primarily sells or does.
    3. Never include the company or brand name.
    4. Describe the campaign's actual creative tactic or execution, not the business problem, strategy, or insight.
    5. Preserve only information explicitly present in the original idea.
    6. Do not infer, embellish, interpret, or add details.
    7. Remove promotional, emotional, and award-entry language.
    8. Prefer concrete nouns and verbs over abstract marketing language.
    9. The sentence must contain either " to " or " with ".
    10. Use "to" when describing the intended purpose or outcome.
    11. Use "with" when describing the mechanism, object, feature, or execution.
    12. Maximum 15 words.
    13. Output exactly one standardized idea.

    Return only valid JSON:
    {"standardized_idea": "..."} """
    prompt = f"Raw idea:\n{raw_idea}"
    return str(gemini_json(prompt, system_instruction, temperature=0.1)["standardized_idea"]).strip()


def generate_idea_paraphrases(standardized_idea: str, n: int = 5) -> List[str]:
    """Generate same-meaning paraphrases using the Gemini model from Exp 2."""
    system_instruction = """You are a strict semantic-preserving paraphrase generator.
Rewrite the supplied idea into unique wording variants that mean exactly the same thing.Have the structure be Subject → Verb → Objectives → Method
Do not add meaning, remove meaning, infer context, broaden the idea, narrow the idea, or make it more creative.
Return only valid JSON with this schema: {"paraphrases": ["..."]}."""
    prompt = f"Create exactly {n} unique paraphrases of this idea:\n{standardized_idea}"
    paraphrases = gemini_json(prompt, system_instruction, temperature=0.4)["paraphrases"]
    return [clean_text(item) for item in paraphrases][:n]


def gemini_embedding(text: str, *, model: str = GEMINI_EMBEDDING_MODEL) -> List[float]:
    """Embed one text with Gemini."""
    cleaned = clean_text(text)
    cache_key = f"{model}::{cleaned}"
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    client = get_gemini_client()
    response = client.models.embed_content(
        model=model,
        contents=cleaned,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )
    vector = list(response.embeddings[0].values)
    _embedding_cache[cache_key] = vector
    return vector


def gemini_embeddings_batch(
    texts: Sequence[str], *, model: str = GEMINI_EMBEDDING_MODEL
) -> List[List[float]]:
    """Embed multiple texts in one Gemini API call, preserving input order.

    Cache hits are served without a network call, same as `gemini_embedding`;
    only the texts not already cached are sent, and as a single batched
    request rather than one request per text. This is the fix for
    `embedding_output` calling `gemini_embedding` in a per-item loop, which
    turned every 5-text batch (4 paraphrases + the original) into 5
    sequential round trips instead of 1.
    """
    cleaned = [clean_text(text) for text in texts]
    results: List[Optional[List[float]]] = [None] * len(cleaned)
    pending_indices: List[int] = []
    pending_texts: List[str] = []

    for i, text in enumerate(cleaned):
        cache_key = f"{model}::{text}"
        cached = _embedding_cache.get(cache_key)
        if cached is not None:
            results[i] = cached
        else:
            pending_indices.append(i)
            pending_texts.append(text)

    if pending_texts:
        client = get_gemini_client()
        response = client.models.embed_content(
            model=model,
            contents=pending_texts,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        if len(response.embeddings) != len(pending_texts):
            raise ValueError(
                "Embedding response length mismatch: expected "
                f"{len(pending_texts)}, got {len(response.embeddings)}"
            )
        for index, text, embedding in zip(pending_indices, pending_texts, response.embeddings):
            vector = list(embedding.values)
            _embedding_cache[f"{model}::{text}"] = vector
            results[index] = vector

    return cast(List[List[float]], results)


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Embedding vector had zero length.")
    return dot / ((norm_a * norm_b) + NORMALIZATION_EPSILON)


def cosine_distance(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Cosine distance between two vectors."""
    return 1.0 - cosine_similarity(vector_a, vector_b)


def gemini_semantic_distance(text_a: str, text_b: str) -> Dict[str, object]:
    """Embed two texts with Gemini, then calculate cosine distance."""
    vector_a = gemini_embedding(text_a)
    vector_b = gemini_embedding(text_b)
    similarity = cosine_similarity(vector_a, vector_b)
    return {
        "semantic_distance": 1.0 - similarity,
        "cosine_similarity": similarity,
        "embedding_model": GEMINI_EMBEDDING_MODEL,
    }


def semantic_distance_between_strings(text_a: str, text_b: str) -> float:
    """Return only the semantic distance number for two strings."""
    return float(gemini_semantic_distance(text_a, text_b)["semantic_distance"])


def _baseline_field_values(
    baseline: Dict[str, str] | pd.DataFrame,
    field: str,
    row_count: int,
) -> List[str]:
    """Return row-aligned baseline values for one field."""
    if isinstance(baseline, pd.DataFrame):
        if field not in baseline.columns:
            raise KeyError(f"Missing baseline column: {field}")
        if len(baseline) == 1:
            return [clean_text(baseline.iloc[0][field])] * row_count
        if len(baseline) == row_count:
            return [clean_text(value) for value in baseline[field].tolist()]
        raise ValueError(
            "Baseline DataFrame must have either one row or the same number "
            f"of rows as ideas. Got baseline={len(baseline)}, ideas={row_count}."
        )

    if field not in baseline:
        raise KeyError(f"Missing baseline field: {field}")
    return [clean_text(baseline[field])] * row_count


def add_semantic_displacement_scores(
    ideas: pd.DataFrame,
    baseline: Dict[str, str] | pd.DataFrame,
) -> pd.DataFrame:
    """Step 2: compare each idea field to the matching baseline field.

    Required DataFrame columns:
    - brand_problem
    - cultural_observation
    - solution_implementation

    Baseline can be:
    - a dict with the same keys
    - a one-row DataFrame with the same columns
    - a same-length DataFrame for row-by-row comparison
    """
    scored = ideas.copy()
    fields = ["brand_problem", "cultural_observation", "solution_implementation"]

    missing_columns = [field for field in fields if field not in scored.columns]
    if missing_columns:
        raise KeyError(f"Missing idea columns: {missing_columns}")

    for field in fields:
        score_column = f"{field}_displacement"
        idea_values = [clean_text(value) for value in scored[field].tolist()]
        baseline_values = _baseline_field_values(baseline, field, len(scored))

        scored[score_column] = [
            semantic_distance_between_strings(idea_text, baseline_text)
            for idea_text, baseline_text in zip(idea_values, baseline_values)
        ]

    scored["semantic_displacement"] = scored[
        [
            "brand_problem_displacement",
            "cultural_observation_displacement",
            "solution_implementation_displacement",
        ]
    ].mean(axis=1)
    return scored


def load_local_causal_lm(model_name: str = CAUSAL_LM_MODEL):
    """Load the local causal LM and tokenizer once."""
    global _local_tokenizer, _local_model, _local_model_name
    if _local_tokenizer is None or _local_model is None or _local_model_name != model_name:
        torch, _ = _load_torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        local_only = Path(model_name).is_absolute()
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=local_only,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = _resolve_device(torch)
        dtype = torch.float16 if device == "cuda" else torch.float32
        # Materialize the checkpoint normally before moving it to the target
        # device.  Meta-tensor loading can leave parameters without storage,
        # which makes model.to(device) fail at runtime in Cloud Run.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            local_files_only=local_only,
        )
        model.to(device)
        model.eval()

        _local_tokenizer = tokenizer
        _local_model = model
        _local_model_name = model_name
    return _local_tokenizer, _local_model


def _build_context_candidate_text(context: str, candidate: str) -> Tuple[str, int]:
    context = clean_text(context)
    candidate = clean_text(candidate)
    if context:
        return f"{context} {candidate}", len(context) + 1
    return candidate, 0


def _tokenize_context_candidate(tokenizer, context: str, candidate: str) -> Dict[str, object]:
    full_text, candidate_start = _build_context_candidate_text(context, candidate)
    candidate_end = len(full_text)
    encoded = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=False,
    )
    candidate_mask = [
        bool(end > candidate_start and start < candidate_end and end > start)
        for start, end in encoded["offset_mapping"]
    ]
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "candidate_mask": candidate_mask,
        "full_text": full_text,
    }


def batch_score_surprise(
    pairs: Sequence[Tuple[str, str]],
    tokenizer=None,
    model=None,
    *,
    model_name: str = CAUSAL_LM_MODEL,
    batch_size: int = SURPRISE_BATCH_SIZE,
) -> List[Dict[str, object]]:
    """Step 3 core: score candidate-token surprise conditioned on context.

    Only candidate tokens contribute to the probability calculation. Context
    tokens are masked out.
    """
    if tokenizer is None or model is None:
        tokenizer, model = load_local_causal_lm(model_name)
    torch, F = _load_torch()
    device = next(model.parameters()).device

    results_by_pair = {}
    pairs_to_score = []
    for context, candidate in pairs:
        cache_key = (model_name, clean_text(context), clean_text(candidate))
        if cache_key in _surprise_cache:
            results_by_pair[cache_key] = _surprise_cache[cache_key]
        else:
            pairs_to_score.append(cache_key)

    for start in range(0, len(pairs_to_score), batch_size):
        batch_keys = pairs_to_score[start:start + batch_size]
        tokenized_items = [
            _tokenize_context_candidate(tokenizer, context, candidate)
            for _, context, candidate in batch_keys
        ]

        max_length = max(len(item["input_ids"]) for item in tokenized_items)
        pad_id = tokenizer.pad_token_id

        input_ids, attention_masks, candidate_masks = [], [], []
        for item in tokenized_items:
            pad_length = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad_length)
            attention_masks.append(item["attention_mask"] + [0] * pad_length)
            candidate_masks.append(item["candidate_mask"] + [False] * pad_length)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long, device=device)
        candidate_masks = torch.tensor(candidate_masks, dtype=torch.bool, device=device)

        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_masks)
            shifted_logits = outputs.logits[:, :-1, :]
            shifted_labels = input_ids[:, 1:]
            shifted_attention = attention_masks[:, 1:].bool()
            shifted_candidate_mask = candidate_masks[:, 1:] & shifted_attention

            log_probs = F.log_softmax(shifted_logits, dim=-1)
            token_log_probs = log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)

            for row_index, cache_key in enumerate(batch_keys):
                mask = shifted_candidate_mask[row_index]
                candidate_token_log_probs = token_log_probs[row_index][mask]
                total_log_probability = float(candidate_token_log_probs.sum().detach().cpu())
                candidate_token_count = int(mask.sum().detach().cpu())
                total_surprise_bits = -total_log_probability / math.log(2)
                average_surprise_bits = total_surprise_bits / max(candidate_token_count, 1)

                candidate_labels = shifted_labels[row_index][mask]
                token_texts = tokenizer.convert_ids_to_tokens(candidate_labels.detach().cpu().tolist())
                token_surprise_bits = (-candidate_token_log_probs / math.log(2)).detach().cpu().tolist()

                score = {
                    "model": model_name,
                    "total_log_probability": total_log_probability,
                    "total_surprise_bits": total_surprise_bits,
                    "average_surprise_bits": average_surprise_bits,
                    "avg_surprise_bits": average_surprise_bits,
                    "candidate_token_count": candidate_token_count,
                    "token_count": candidate_token_count,
                    "tokens": [
                        {"text": text, "surprise_bits": float(bits)}
                        for text, bits in zip(token_texts, token_surprise_bits)
                    ],
                }
                _surprise_cache[cache_key] = score
                results_by_pair[cache_key] = score

    return [
        results_by_pair[(model_name, clean_text(context), clean_text(candidate))]
        for context, candidate in pairs
    ]


def score_candidate_surprise(
    context: str,
    candidate: str,
    tokenizer=None,
    model=None,
    *,
    model_name: str = CAUSAL_LM_MODEL,
) -> Dict[str, object]:
    """Convenience wrapper for one context/candidate pair."""
    return batch_score_surprise(
        [(context, candidate)],
        tokenizer=tokenizer,
        model=model,
        model_name=model_name,
        batch_size=1,
    )[0]


def score_surprise(context: str, candidate: str, *, model_name: str = CAUSAL_LM_MODEL) -> Dict[str, object]:
    """Alias used by Exp 2."""
    return score_candidate_surprise(context, candidate, model_name=model_name)


def add_contextual_surprise_scores(
    ideas: pd.DataFrame,
    tokenizer=None,
    model=None,
    *,
    model_name: str = CAUSAL_LM_MODEL,
    use_average_surprise: bool = USE_AVERAGE_SURPRISE,
) -> pd.DataFrame:
    """Step 3: add observation and solution contextual surprise scores."""
    scored = ideas.copy()
    fields = ["brand_problem", "cultural_observation", "solution_implementation"]
    missing_columns = [field for field in fields if field not in scored.columns]
    if missing_columns:
        raise KeyError(f"Missing idea columns: {missing_columns}")

    observation_pairs = list(zip(scored["brand_problem"], scored["cultural_observation"]))
    solution_contexts = [
        f"brand problem: {row.brand_problem} cultural observation: {row.cultural_observation} solution:"
        for row in scored.itertuples(index=False)
    ]
    solution_pairs = list(zip(solution_contexts, scored["solution_implementation"]))

    observation_scores = batch_score_surprise(
        observation_pairs,
        tokenizer=tokenizer,
        model=model,
        model_name=model_name,
    )
    solution_scores = batch_score_surprise(
        solution_pairs,
        tokenizer=tokenizer,
        model=model,
        model_name=model_name,
    )

    for prefix, scores in [("observation", observation_scores), ("solution", solution_scores)]:
        scored[f"{prefix}_total_surprise_bits"] = [score["total_surprise_bits"] for score in scores]
        scored[f"{prefix}_avg_surprise_bits"] = [score["average_surprise_bits"] for score in scores]
        scored[f"{prefix}_candidate_token_count"] = [score["candidate_token_count"] for score in scores]

    observation_col = "observation_avg_surprise_bits" if use_average_surprise else "observation_total_surprise_bits"
    solution_col = "solution_avg_surprise_bits" if use_average_surprise else "solution_total_surprise_bits"
    scored["contextual_surprise"] = (scored[observation_col] + scored[solution_col]) / 2
    return scored


def add_relationship_incongruity_scores(ideas: pd.DataFrame) -> pd.DataFrame:
    """Step 4: score whether the strategic structure hangs together.

    observation_problem_incongruity =
        distance(E(cultural_observation), E(brand_problem))

    solution_context_incongruity =
        distance(E(solution_implementation), E(brand_problem + cultural_observation))

    internal_incongruity =
        mean(observation_problem_incongruity, solution_context_incongruity)
    """
    scored = ideas.copy()
    fields = ["brand_problem", "cultural_observation", "solution_implementation"]
    missing_columns = [field for field in fields if field not in scored.columns]
    if missing_columns:
        raise KeyError(f"Missing idea columns: {missing_columns}")

    problem_texts = [clean_text(text) for text in scored["brand_problem"].tolist()]
    observation_texts = [clean_text(text) for text in scored["cultural_observation"].tolist()]
    solution_texts = [clean_text(text) for text in scored["solution_implementation"].tolist()]
    problem_observation_contexts = [
        clean_text(f"Brand problem: {row.brand_problem} Cultural observation: {row.cultural_observation}")
        for row in scored.itertuples(index=False)
    ]

    observation_problem_scores = []
    solution_context_scores = []
    for problem, observation, solution, context in zip(
        problem_texts,
        observation_texts,
        solution_texts,
        problem_observation_contexts,
    ):
        observation_problem_scores.append(
            cosine_distance(gemini_embedding(observation), gemini_embedding(problem))
        )
        solution_context_scores.append(
            cosine_distance(gemini_embedding(solution), gemini_embedding(context))
        )

    scored["observation_problem_incongruity"] = observation_problem_scores
    scored["solution_context_incongruity"] = solution_context_scores
    scored["internal_incongruity"] = scored[
        ["observation_problem_incongruity", "solution_context_incongruity"]
    ].mean(axis=1)
    return scored


def relationship_incongruity_between_strings(
    brand_problem: str,
    cultural_observation: str,
    solution_implementation: str,
) -> Dict[str, float]:
    """Step 4 for one idea passed as strings.

    Returns the two relationship distances and their mean:
    - cultural observation vs brand problem
    - solution implementation vs brand problem + cultural observation
    """
    problem = clean_text(brand_problem)
    observation = clean_text(cultural_observation)
    solution = clean_text(solution_implementation)
    context = clean_text(f"Brand problem: {problem} Cultural observation: {observation}")
    context = problem


    observation_problem_incongruity = semantic_distance_between_strings(observation, problem)
    solution_context_incongruity = semantic_distance_between_strings(solution, context)
    internal_incongruity = (
        observation_problem_incongruity + solution_context_incongruity
    ) / 2

    return {
        "observation_problem_incongruity": observation_problem_incongruity,
        "solution_context_incongruity": solution_context_incongruity,
        "internal_incongruity": internal_incongruity,
    }


def add_incongruity_scores(ideas: pd.DataFrame, *_, **__) -> pd.DataFrame:
    """Pipeline-compatible wrapper for the Step 4 relationship metric."""
    return add_relationship_incongruity_scores(ideas)


def validate_campaign_components(result: Dict[str, object], target_words: int) -> Dict[str, str]:
    """Validate component extractor output shape and word counts."""
    required = ["cultural_observation", "problem_to_solve", "solution"]
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    cleaned = {key: clean_text(result[key]) for key in required}
    counts = {key: count_words(value) for key, value in cleaned.items()}
    bad = {key: value for key, value in counts.items() if value != target_words}
    if bad:
        raise ValueError(f"Each field must have exactly {target_words} words. Got: {counts}")
    if not cleaned["problem_to_solve"].startswith("A brand is trying to"):
        raise ValueError("problem_to_solve must start with: A brand is trying to")

    contrastive_terms = [
        "while",
        "but",
        "yet",
        "although",
        "despite",
        "even though",
        "however",
    ]
    observation_lower = cleaned["cultural_observation"].lower()
    blocked_terms = [
        term
        for term in contrastive_terms
        if re.search(rf"\b{re.escape(term)}\b", observation_lower)
    ]
    if blocked_terms:
        raise ValueError(
            "cultural_observation must be atomic and cannot use contrastive terms. "
            f"Found: {blocked_terms}"
        )
    return cleaned


def extract_campaign_components(
    campaign: str,
    *,
    target_words: int = 18,
    use_web_search: bool = True,
    model: Optional[str] = None,
    max_retries: int = 3,
) -> Dict[str, str]:
    """Extract cultural observation, problem to solve, and solution from a campaign."""
    model = model or GEMINI_MODEL
    system_instruction = f"""You extract normalized strategic components from advertising campaigns.

Return exactly three fields as valid JSON:
- cultural_observation
- problem_to_solve
- solution

Rules:
- Every field must have exactly {target_words} words.
- Every field must use the same plain, literal, analytical tone.
- Structure every problem and solution as Subject → Verb → Objectives → Method
- Every field must use simple present-tense business language.
- Do not use hype, awards language, poetic language, or creative taglines.
- Do not mention sources or uncertainty in the JSON.
- cultural_observation must be a single, atomic cultural observation.
- cultural_observation must state only one observable behavior, habit, norm, or cultural truth.
- cultural_observation must not introduce a contrast, tension, problem, consequence, or opportunity.
- cultural_observation must not use contrastive constructions such as while, but, yet, although, despite, even though, or however.
- cultural_observation must not explain why the observation matters.
- cultural_observation must not connect the behavior to the brand problem or solution.
- cultural_observation must end immediately after stating the observation.
- problem_to_solve explains what the business is trying to achieve.
- problem_to_solve must start exactly with: A brand is trying to
- solution explains how the problem is solved through the campaign idea.
- If web search is available, use it only to understand the campaign accurately.
- Return only valid JSON, with no markdown."""

    prompt = f"""Campaign:
{campaign}

Extract the three standardized fields. Make all three fields exactly {target_words} words."""

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = gemini_json(
                prompt,
                system_instruction,
                model=model,
                temperature=0.1,
                use_web_search=use_web_search,
                max_output_tokens=4096,
            )
            return validate_campaign_components(result, target_words)
        except ValueError as exc:
            last_error = exc
            prompt = f"""Campaign:
{campaign}

Your last response failed validation: {exc}

Return valid JSON again. Every field must have exactly {target_words} words."""
            if attempt == max_retries:
                raise

    raise RuntimeError(f"Gemini did not return a valid result: {last_error}")


def embedding_output(list_value: Sequence[str]) -> List[List[float]]:
    """Embed a list of phrases with Gemini in a single batched call."""
    return gemini_embeddings_batch(list_value)


def preprocessing_value_generation(
    cultural_observation: str,
    problem_to_solve: str,
    solution: str,
    *,
    paraphrase_count: int = 4,
) -> Dict[str, object]:
    """Create one fixed paraphrase and embedding bundle for an idea.

    The original phrase is appended to the generated paraphrases, so each field
    has `paraphrase_count + 1` variants.
    """
    cultural_observation_list = (
        generate_idea_paraphrases(cultural_observation, n=paraphrase_count)
        + [clean_text(cultural_observation)]
    )
    problem_to_solve_list = (
        generate_idea_paraphrases(problem_to_solve, n=paraphrase_count)
        + [clean_text(problem_to_solve)]
    )
    solution_list = (
        generate_idea_paraphrases(solution, n=paraphrase_count)
        + [clean_text(solution)]
    )

    return {
        "input": {
            "cultural_observation": clean_text(cultural_observation),
            "problem_to_solve": clean_text(problem_to_solve),
            "solution": clean_text(solution),
        },
        "paraphrases": {
            "cultural_observation": cultural_observation_list,
            "problem_to_solve": problem_to_solve_list,
            "solution": solution_list,
        },
        "embeddings": {
            "cultural_observation": embedding_output(cultural_observation_list),
            "problem_to_solve": embedding_output(problem_to_solve_list),
            "solution": embedding_output(solution_list),
        },
    }


def select_baseline(embeddings: Sequence[Sequence[float]]) -> int:
    """Select the most central paraphrase embedding from a list."""
    if not embeddings:
        raise ValueError("Cannot select a baseline from an empty embedding list.")
    if len(embeddings) == 1:
        return 0

    scores = []
    for i, embedding in enumerate(embeddings):
        similarities = [
            cosine_similarity(embedding, other_embedding)
            for j, other_embedding in enumerate(embeddings)
            if i != j
        ]
        scores.append(sum(similarities) / len(similarities))
    return max(range(len(scores)), key=lambda index: scores[index])


def select_baseline_full(prepped_baseline: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    """Select one central baseline phrase and embedding for each component."""
    keys = ["cultural_observation", "problem_to_solve", "solution"]
    selected = {}

    for key in keys:
        index = select_baseline(prepped_baseline["embeddings"][key])
        selected[key] = {
            "phrase": prepped_baseline["paraphrases"][key][index],
            "embedding": prepped_baseline["embeddings"][key][index],
        }

    return selected


def build_baseline_dict(
    cultural_observation: str = "A person wants to buy a product",
    problem_to_solve: str = "A brand needs to sell a product",
    solution: str = "The brand markets to a person trying to buy a product",
    *,
    paraphrase_count: int = 4,
) -> Dict[str, Dict[str, object]]:
    """Create a baseline dictionary with selected central paraphrases."""
    prepped_baseline = preprocessing_value_generation(
        cultural_observation=cultural_observation,
        problem_to_solve=problem_to_solve,
        solution=solution,
        paraphrase_count=paraphrase_count,
    )
    return select_baseline_full(prepped_baseline)


def save_baseline_dict(baseline: Dict[str, object], path: str | Path) -> None:
    """Save a baseline dictionary locally as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")


def load_baseline_dict(path: str | Path) -> Dict[str, object]:
    """Load a saved baseline dictionary from JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_or_create_baseline_dict(
    path: str | Path,
    *,
    cultural_observation: str = "A person wants to buy a product",
    problem_to_solve: str = "A brand needs to sell a product",
    solution: str = "The brand markets to a person trying to buy a product",
    paraphrase_count: int = 4,
    force_recreate: bool = False,
) -> Dict[str, object]:
    """Load a local baseline JSON, or create and save it once if missing."""
    path = Path(path)
    if path.exists() and not force_recreate:
        return load_baseline_dict(path)

    baseline = build_baseline_dict(
        cultural_observation=cultural_observation,
        problem_to_solve=problem_to_solve,
        solution=solution,
        paraphrase_count=paraphrase_count,
    )
    save_baseline_dict(baseline, path)
    return baseline


def distance_from_baseline(
    baseline: Dict[str, Dict[str, object]],
    campaign: Dict[str, object],
) -> Dict[str, Dict[str, float]]:
    """Compare campaign paraphrase embeddings to the selected baseline embeddings."""
    keys = ["cultural_observation", "problem_to_solve", "solution"]
    results = {}

    for key in keys:
        baseline_embedding = baseline[key]["embedding"]
        campaign_embeddings = campaign["embeddings"][key]
        distances = [
            cosine_distance(baseline_embedding, embedding)
            for embedding in campaign_embeddings
        ]
        results[key] = summarize_values(distances)

    return results


def distance_from_multiple_baselines(
    baselines: Dict[str, Dict[str, Dict[str, object]]] | Sequence[Dict[str, Dict[str, object]]],
    campaign: Dict[str, object],
    *,
    metric: str = "mean",
) -> Dict[str, object]:
    """Compare one campaign to many saved baseline dictionaries.

    `baselines` can be either:
    - {"baseline_name": baseline_dict, ...}
    - [baseline_dict_1, baseline_dict_2, ...]

    Each `baseline_dict` should match the saved baseline format:
    baseline["cultural_observation"]["embedding"]
    baseline["problem_to_solve"]["embedding"]
    baseline["solution"]["embedding"]

    The returned `distance_score` is one measurement: the average of each
    baseline's weighted distance score.
    """
    if isinstance(baselines, dict):
        baseline_items = list(baselines.items())
    else:
        baseline_items = [
            (f"baseline_{index + 1}", baseline)
            for index, baseline in enumerate(baselines)
        ]

    if not baseline_items:
        raise ValueError("At least one baseline is required.")

    baseline_results = {}
    distance_scores = []

    for baseline_name, baseline in baseline_items:
        if not is_single_baseline_dict(baseline):
            keys = list(baseline.keys()) if isinstance(baseline, dict) else type(baseline).__name__
            raise ValueError(
                f"Baseline `{baseline_name}` is not a full baseline dict. "
                "Expected keys: cultural_observation, problem_to_solve, solution, "
                f"each containing an embedding. Got: {keys}"
            )
        distance = distance_from_baseline(baseline, campaign)
        distance_score = calculate_weighted_score_distance(distance, metric=metric)
        baseline_results[baseline_name] = {
            "distance": distance,
            "distance_score": distance_score,
        }
        distance_scores.append(distance_score)

    summary = summarize_values(distance_scores)
    return {
        "distance_score": summary["mean"],
        "distance_score_median": summary["median"],
        "distance_score_std": summary["std"],
        "distance_scores": distance_scores,
        "baseline_results": baseline_results,
    }


def is_single_baseline_dict(value: object) -> bool:
    """Return True when value matches one saved baseline dictionary."""
    if not isinstance(value, dict):
        return False
    required_keys = {"cultural_observation", "problem_to_solve", "solution"}
    if not required_keys.issubset(value.keys()):
        return False
    return all(
        isinstance(value[key], dict) and "embedding" in value[key]
        for key in required_keys
    )


def summarize_values(values: Sequence[float]) -> Dict[str, float]:
    """Return mean, median, std, and raw values for a numeric sequence."""
    if not values:
        return {"mean": None, "median": None, "std": None, "values": []}

    sorted_values = sorted(float(value) for value in values)
    count = len(sorted_values)
    midpoint = count // 2
    if count % 2:
        median = sorted_values[midpoint]
    else:
        median = (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2

    mean = sum(sorted_values) / count
    variance = sum((value - mean) ** 2 for value in sorted_values) / count
    return {
        "mean": mean,
        "median": median,
        "std": math.sqrt(variance),
        "values": sorted_values,
    }


def calculate_weighted_score_distance(
    scores: Dict[str, Dict[str, float]],
    metric: str = "mean",
) -> float:
    """Calculate weighted semantic distance score."""
    weights = {
        "cultural_observation": 2,
        "problem_to_solve": 1,
        "solution": 2,
    }
    weighted_sum = sum(scores[key][metric] * weight for key, weight in weights.items())
    return weighted_sum / sum(weights.values())


def calc_surprise(
    campaignfull: Dict[str, object],
    *,
    context: str = "An example of a marketing campaign is",
) -> Dict[str, object]:
    """Summarize Qwen surprise across the solution paraphrases."""
    surprise = [
        score_surprise(context=context, candidate=idea.lower())["avg_surprise_bits"]
        for idea in campaignfull["paraphrases"]["solution"]
    ]
    summary = summarize_values(surprise)
    summary["scores"] = surprise
    summary["solutions"] = campaignfull["paraphrases"]["solution"]
    return summary


def score_solution_surprise(
    solution: str,
    *,
    paraphrase_count: int = 4,
    context: str = "An example of a marketing campaign is",
) -> Dict[str, object]:
    """Score one solution using the same method as the main surprise score.

    The original solution plus semantic-preserving paraphrases are scored with
    Qwen, then averaged so the result is less dependent on one exact wording.
    """
    solutions = generate_idea_paraphrases(solution, n=paraphrase_count) + [clean_text(solution)]
    scores = [
        score_surprise(context=context, candidate=variant.lower())["avg_surprise_bits"]
        for variant in solutions
    ]
    summary = summarize_values(scores)
    summary["solution"] = clean_text(solution)
    summary["variants"] = solutions
    summary["scores"] = scores
    return summary


def generate_surprise_thoughtstarters(
    cultural_observation: str,
    brand_problem: str,
    *,
    count: int = 5,
) -> Dict[str, List[str]]:
    """Generate conventional and deliberately chaotic solutions.

    Both sets use the same observation and problem. Because this endpoint does
    not receive an existing solution, "more chaotic" means more unexpected,
    indirect, and creatively disproportionate than a conventional solution;
    "less chaotic" means direct, familiar, and easy to predict.
    """
    system_instruction = """You generate advertising campaign solution thoughtstarters.

Return only valid JSON with exactly this schema:
{"more_chaotic": ["..."], "less_chaotic": ["..."]}

Generate exactly the requested number of unique solutions in each list.
Every solution must describe a concrete campaign action or mechanism that a
brand could execute. Keep each solution to one concise sentence.

more_chaotic solutions should be surprising, indirect, strange, or
unexpectedly connected to the observation and problem while still being
coherent and executable.
less_chaotic solutions should be direct, familiar, conventional, and easy to
predict from the observation and problem.

Do not explain the ideas, add headings, or include scores."""
    prompt = f"""Cultural observation:
{clean_text(cultural_observation)}

Brand problem:
{clean_text(brand_problem)}

Generate exactly {count} more chaotic and exactly {count} less chaotic solutions."""
    result = gemini_json(prompt, system_instruction, temperature=0.8, max_output_tokens=4096)

    output: Dict[str, List[str]] = {}
    for key in ("more_chaotic", "less_chaotic"):
        values = result.get(key)
        if not isinstance(values, list):
            raise ValueError(f"Gemini response field `{key}` must be a list.")
        cleaned = [clean_text(value) for value in values if isinstance(value, str) and clean_text(value)]
        if len(cleaned) < count:
            raise ValueError(f"Gemini returned fewer than {count} `{key}` solutions.")
        output[key] = cleaned[:count]
    return output


def calculate_incongruity(idea: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    """Calculate pairwise component tension across matching paraphrase embeddings."""
    embeddings = idea["embeddings"]
    pairs = {
        "cultural_observation_solution": ("cultural_observation", "solution"),
        "solution_problem_to_solve": ("solution", "problem_to_solve"),
        "problem_to_solve_cultural_observation": ("problem_to_solve", "cultural_observation"),
    }
    results = {}

    for name, (key_a, key_b) in pairs.items():
        distances = [
            cosine_distance(a, b)
            for a, b in zip(embeddings[key_a], embeddings[key_b])
        ]
        results[name] = summarize_values(distances)

    return results


def calculate_weighted_score_tension(
    scores: Dict[str, Dict[str, float]],
    metric: str = "mean",
) -> float:
    """Calculate weighted internal tension score."""
    weights = {
        "cultural_observation_solution": 3,
        "solution_problem_to_solve": 0.5,
        "problem_to_solve_cultural_observation": 2,
    }
    weighted_sum = sum(scores[key][metric] * weight for key, weight in weights.items())
    return weighted_sum / sum(weights.values())



def predict_basicness_from_scores(
    distance_score: float,
    surprise_score: float,
    text: str,
    path: str | Path = "basicness_detector_model.pkl",
) -> Tuple[bool, float]:
    """Return whether a campaign is basic and the model probability.

    High-confidence model calls are handled directly. Borderline model calls
    are adjudicated by Gemini using the supplied idea text.
    """
    artifact = load_saved_basicness_model(path)

    raw = np.array([float(distance_score), float(surprise_score)], dtype=float)
    if np.isnan(raw).any():
        impute_values = np.array(artifact["impute_values"], dtype=float)
        raw = np.where(np.isnan(raw), impute_values, raw)

    z = (
        raw - np.array(artifact["scale_mean"], dtype=float)
    ) / np.array(artifact["scale_scale"], dtype=float)

    log_odds = float(
        np.dot(z, np.array(artifact["coef"], dtype=float))
        + float(artifact["intercept"])
    )
    probability = float(1 / (1 + np.exp(-log_odds)))

    if probability >= 0.95:
        return True, probability

    
    if probability > 0.49:

        system_instruction ="""
            You are a marketing idea detector that determines whether a creative marketing idea is BASIC.

            A BASIC idea:
            - Uses common, obvious, or conventional marketing tactics.
            - Has little or no distinctive creative use of a tactic.

            Return only valid JSON using exactly this schema:
            {"Basic": true}

            or

            {"Basic": false}"""
                    
        values = gemini_json(text, system_instruction, temperature=0.1)
        return bool(values["Basic"]), probability
    return False, probability


def normalize_basicness_result(result: object) -> Tuple[bool, float]:
    """Accept old and new basicness return shapes."""
    if isinstance(result, (list, tuple)):
        if len(result) < 2:
            raise ValueError("Basicness result must include basic flag and probability.")
        return bool(result[0]), float(result[1])
    return bool(result), 1.0 if bool(result) else 0.0




def universal_score(distance, surprise, tension,basic,prob):
    penalize = 1 #no pentality
    if basic == True:
        if prob > .9:
            penalize = .75
        else: #smaller penality
            penalize = .9

    distance_threshold = 0.21
    surprise_threshold = 9.389214
    tension_threshold = 0.192203

    # Normalize relative to threshold
    d = distance*penalize / distance_threshold
    s = surprise*penalize / surprise_threshold
    t = tension*penalize / tension_threshold

    # Average performance
    avg_score = (d + s + t) / 3

    # Reward one exceptionally high dimension
    max_score = max(d, s, t)

    # Combined universal score
    score = (0.8 * avg_score) + (0.2 * max_score)

    return score * 100

def full_analysis(
    campaign: str,
    baseline: Dict[str, Dict[str, object]] | Dict[str, Dict[str, Dict[str, object]]] | Sequence[Dict[str, Dict[str, object]]],
    *,
    target_words: int = 14,
    use_web_search: bool = True,
    paraphrase_count: int = 4,
) -> Dict[str, object]:
    """Standardize one campaign idea, extract components, and score it."""
    analysis_started = time.perf_counter()
    stage_started = time.perf_counter()
    standardized_idea = standardize_idea(campaign)
    logger.info("score_timing stage=standardize_idea seconds=%.3f", time.perf_counter() - stage_started)

    stage_started = time.perf_counter()
    components_idea = extract_campaign_components(
        standardized_idea,
        target_words=target_words,
        use_web_search=use_web_search,
    )
    logger.info("score_timing stage=extract_components seconds=%.3f", time.perf_counter() - stage_started)

    stage_started = time.perf_counter()
    campaign_embed_paraphrases = preprocessing_value_generation(
        cultural_observation=components_idea["cultural_observation"],
        problem_to_solve=components_idea["problem_to_solve"],
        solution=components_idea["solution"],
        paraphrase_count=paraphrase_count,
    )
    logger.info("score_timing stage=paraphrases_and_embeddings seconds=%.3f", time.perf_counter() - stage_started)

    stage_started = time.perf_counter()
    distance = distance_from_multiple_baselines(baseline, campaign_embed_paraphrases)
    distance_score = distance["distance_score"]
    distance_std= distance['distance_score_std']
    logger.info("score_timing stage=distance seconds=%.3f", time.perf_counter() - stage_started)

    stage_started = time.perf_counter()
    surprise = calc_surprise(campaign_embed_paraphrases)
    surprise_score = surprise["mean"]
    surprise_std = surprise["std"]
    logger.info("score_timing stage=surprise seconds=%.3f", time.perf_counter() - stage_started)

    stage_started = time.perf_counter()
    tension = calculate_incongruity(campaign_embed_paraphrases)
    tension_score = calculate_weighted_score_tension(tension)
    tension_std = calculate_weighted_score_tension(tension,'std')
    logger.info("score_timing stage=tension seconds=%.3f", time.perf_counter() - stage_started)

    stage_started = time.perf_counter()
    basic_score_result = predict_basicness_from_scores(distance_score=distance_score,
                                                surprise_score=surprise_score,
                                                text=standardized_idea)

    basic_score, probability = normalize_basicness_result(basic_score_result)
    logger.info("score_timing stage=basicness seconds=%.3f", time.perf_counter() - stage_started)

    universal_score_high = universal_score(distance_score+distance_std*2, 
                                           surprise_score+surprise_std*2,
                                           tension_score+tension_std*2, 
                                           basic_score,probability)
    
    universal_score_ = universal_score(distance_score, surprise_score, tension_score, basic_score,probability)

    universal_score_low = universal_score(distance_score-distance_std*2, 
                                           surprise_score-surprise_std*2,
                                           tension_score-tension_std*2, 
                                           basic_score,probability)
    logger.info("score_timing stage=total seconds=%.3f", time.perf_counter() - analysis_started)
    
    return {
        "parts": {
            "idea": campaign,
            "standardized_idea": standardized_idea,
            "components_idea": components_idea,
            "campaign_embed_paraphrases": campaign_embed_paraphrases,
            "distance": distance,
            "surprise": surprise,
            "tension": tension,
            'basic':[basic_score,probability]

        },
        "scores": {
            "distance_score": distance_score,
            "surprise_score": surprise_score,
            "tension_score": tension_score,
            'basic_score':basic_score,
            'universal_score':universal_score_,
            'universal_score_range':[universal_score_low,universal_score_high],
            'std':[distance_std,surprise_std,tension_std]

        },
    }


def flatten_prior_art_batches_to_csv(
    batches_dir: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Flatten prior-art batch JSON files into a CSV with an `idea` column."""
    batches_dir = Path(batches_dir)
    rows = []

    for path in sorted(batches_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else [data]

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            idea_parts = [
                record.get("name", ""),
                record.get("insight", ""),
                record.get("mechanism", ""),
                record.get("output", ""),
            ]
            rows.append({
                "source_file": path.name,
                "source_index": index,
                "name": clean_text(record.get("name", "")),
                "input": clean_text(record.get("input", "")),
                "mechanism": clean_text(record.get("mechanism", "")),
                "insight": clean_text(record.get("insight", "")),
                "output": clean_text(record.get("output", "")),
                "format": clean_text(record.get("format", "")),
                "market": clean_text(record.get("market", "")),
                "source": clean_text(record.get("source", "")),
                "notes": clean_text(record.get("notes", "")),
                "idea": clean_text(" ".join(str(part) for part in idea_parts if part)),
            })

    df = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def flatten_analysis_result(result: Dict[str, object]) -> Dict[str, object]:
    """Flatten a full_analysis result into one CSV-friendly row."""
    parts = result.get("parts", {})
    scores = result.get("scores", {})
    components = parts.get("components_idea", {})

    return {
        "standardized_idea": parts.get("standardized_idea"),
        "cultural_observation": components.get("cultural_observation"),
        "problem_to_solve": components.get("problem_to_solve"),
        "solution": components.get("solution"),
        "distance_score": scores.get("distance_score"),
        "surprise_score": scores.get("surprise_score"),
        "tension_score": scores.get("tension_score"),
        "basic_score": scores.get("basic_score"),
        "universal_score": scores.get("universal_score"),
        "distance_json": json.dumps(parts.get("distance", {})),
        "surprise_json": json.dumps(parts.get("surprise", {})),
        "tension_json": json.dumps(parts.get("tension", {})),
        "full_result_json": json.dumps(result),
    }


def score_ideas_csv_with_progress(
    input_csv: str | Path,
    baseline: Dict[str, Dict[str, object]],
    output_csv: str | Path,
    *,
    idea_column: str = "idea",
    limit: Optional[int] = None,
    target_words: int = 14,
    use_web_search: bool = True,
    paraphrase_count: int = 4,
    print_progress: bool = True,
) -> pd.DataFrame:
    """Score ideas from a CSV and save progress after every row."""
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(input_csv)
    if idea_column not in source_df.columns:
        raise KeyError(f"Input CSV must contain an `{idea_column}` column.")
    if limit is not None:
        source_df = source_df.head(limit).copy()

    if output_csv.exists():
        scored_df = pd.read_csv(output_csv)
        completed_indices = set(scored_df["source_row_index"].astype(int).tolist())
        rows = scored_df.to_dict("records")
    else:
        completed_indices = set()
        rows = []

    for source_row_index, row in source_df.iterrows():
        if int(source_row_index) in completed_indices:
            if print_progress:
                print(f"Skipping row {source_row_index + 1}/{len(source_df)}: already scored")
            continue

        output_row = row.to_dict()
        output_row["source_row_index"] = int(source_row_index)
        output_row.update({
            "standardized_idea": None,
            "cultural_observation": None,
            "problem_to_solve": None,
            "solution": None,
            "distance_score": None,
            "surprise_score": None,
            "tension_score": None,
            "basic_score": None,
            "universal_score": None,
            "distance_json": None,
            "surprise_json": None,
            "tension_json": None,
            "full_result_json": None,
        })
        try:
            result = full_analysis(
                str(row[idea_column]),
                baseline,
                target_words=target_words,
                use_web_search=use_web_search,
                paraphrase_count=paraphrase_count,
            )
            output_row.update(flatten_analysis_result(result))
            output_row["status"] = "ok"
            output_row["error"] = ""
            if print_progress:
                print(f"Scored row {source_row_index + 1}/{len(source_df)}: {row.get('name', '')}")
        except Exception as exc:
            output_row["status"] = "error"
            output_row["error"] = f"{type(exc).__name__}: {exc}"
            if print_progress:
                print(f"Error row {source_row_index + 1}/{len(source_df)}: {row.get('name', '')} - {exc}")

        rows.append(output_row)
        pd.DataFrame(rows).to_csv(output_csv, index=False)

    return pd.DataFrame(rows)


def min_max_normalize_series(series: pd.Series) -> pd.Series:
    """Normalize numeric values to 0-1, keeping missing values as missing."""
    numeric = pd.to_numeric(series, errors="coerce")
    minimum = numeric.min()
    maximum = numeric.max()
    if pd.isna(minimum) or pd.isna(maximum):
        return numeric
    if math.isclose(maximum, minimum):
        return numeric.apply(lambda value: 0.5 if pd.notna(value) else value)
    return (numeric - minimum) / (maximum - minimum)


def _short_hover_value(value: object, max_chars: int = 220) -> str:
    """Format a DataFrame value for Plotly hover text."""
    if value is None or pd.isna(value):
        return ""
    text = clean_text(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _build_hover_rows(
    df: pd.DataFrame,
    hover_columns: Sequence[str],
) -> List[str]:
    """Build HTML hover blocks from selected DataFrame columns."""
    rows = []
    for _, row in df.iterrows():
        parts = []
        for column in hover_columns:
            if column not in df.columns:
                continue
            value = _short_hover_value(row[column])
            if not value:
                continue
            label = column.replace("_", " ")
            parts.append(f"<b>{label}</b>: {value}")
        rows.append("<br>".join(parts))
    return rows


def plot_chaos_metric_space(
    results: pd.DataFrame,
    x_axis: str = "normalized_displacement",
    y_axis: str = "normalized_surprise",
    z_axis: str = "normalized_incongruity",
    hover_columns: Optional[Sequence[str]] = None,
):
    """Create an interactive 3D scatter plot across the three chaos-score axes.

    This version is adapted to the current scoring output:
    - distance_score -> normalized_displacement
    - surprise_score -> normalized_surprise
    - tension_score -> normalized_incongruity

    Plotly renders an interactive graph in Jupyter: drag rotates the 3D view,
    scroll/pinch zooms, and hover shows idea-level metrics.
    """
    import numpy as np
    import plotly.graph_objects as go

    plot_df = results.copy()
    if "status" in plot_df.columns:
        plot_df = plot_df[plot_df["status"].fillna("ok").eq("ok")].copy()

    required_raw = ["distance_score", "surprise_score", "tension_score"]
    missing_raw = [column for column in required_raw if column not in plot_df.columns]
    if missing_raw:
        raise KeyError(f"Missing required score columns: {missing_raw}")

    if "normalized_displacement" not in plot_df.columns:
        plot_df["normalized_displacement"] = min_max_normalize_series(plot_df["distance_score"])
    if "normalized_surprise" not in plot_df.columns:
        plot_df["normalized_surprise"] = min_max_normalize_series(plot_df["surprise_score"])
    if "normalized_incongruity" not in plot_df.columns:
        plot_df["normalized_incongruity"] = min_max_normalize_series(plot_df["tension_score"])
    if "chaos_score" not in plot_df.columns:
        plot_df["chaos_score"] = plot_df[
            ["normalized_displacement", "normalized_surprise", "normalized_incongruity"]
        ].mean(axis=1)

    for column in [x_axis, y_axis, z_axis, "chaos_score"]:
        if column not in plot_df.columns:
            raise KeyError(f"Missing plot column: {column}")

    label_col = "name" if "name" in plot_df.columns else "idea"
    if label_col not in plot_df.columns:
        plot_df["idea"] = [f"Idea {index + 1}" for index in range(len(plot_df))]
        label_col = "idea"

    reference_col = "reference_label"
    if reference_col not in plot_df.columns:
        plot_df[reference_col] = "scored idea"
        if "source_file" in plot_df.columns:
            plot_df.loc[plot_df["source_file"].eq("basic_marketing_ideas"), reference_col] = "basic marketing idea"

    plot_df = plot_df.dropna(subset=[x_axis, y_axis, z_axis, "chaos_score"]).copy()
    default_hover_columns = [
        "name",
        "idea",
        "standardized_idea",
        "cultural_observation",
        "problem_to_solve",
        "solution",
        "distance_score",
        "surprise_score",
        "tension_score",
        "basic_score",
        "universal_score",
        "status",
    ]
    hover_columns = list(hover_columns or default_hover_columns)
    plot_df["hover_row_data"] = _build_hover_rows(plot_df, hover_columns)

    regular_rows = plot_df[plot_df[reference_col].eq("scored idea")]
    reference_rows = plot_df[~plot_df[reference_col].eq("scored idea")]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=regular_rows[x_axis],
        y=regular_rows[y_axis],
        z=regular_rows[z_axis],
        mode="markers",
        name="prior-art campaigns",
        text=regular_rows[label_col],
        customdata=np.stack([
            regular_rows["chaos_score"],
            regular_rows["distance_score"],
            regular_rows["surprise_score"],
            regular_rows["tension_score"],
            regular_rows["hover_row_data"],
        ], axis=-1) if len(regular_rows) else np.empty((0, 5)),
        marker=dict(
            size=5,
            color=regular_rows["chaos_score"],
            colorscale="Viridis",
            opacity=0.72,
            colorbar=dict(title="Chaos score"),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "normalized displacement: %{x:.3f}<br>"
            "normalized surprise: %{y:.3f}<br>"
            "normalized incongruity: %{z:.3f}<br>"
            "chaos score: %{customdata[0]:.3f}<br>"
            "distance score: %{customdata[1]:.3f}<br>"
            "surprise score: %{customdata[2]:.3f}<br>"
            "tension score: %{customdata[3]:.3f}<br>"
            "<br>%{customdata[4]}"
            "<extra></extra>"
        ),
    ))

    symbol_by_label = {
        "basic marketing idea": "diamond",
        "primitive baseline": "x",
        "original phrase": "diamond",
    }
    color_by_label = {
        "basic marketing idea": "orange",
        "primitive baseline": "red",
        "original phrase": "blue",
    }

    for label, rows_for_label in reference_rows.groupby(reference_col):
        fig.add_trace(go.Scatter3d(
            x=rows_for_label[x_axis],
            y=rows_for_label[y_axis],
            z=rows_for_label[z_axis],
            mode="markers",
            name=str(label),
            text=rows_for_label[label_col],
            customdata=np.stack([
                rows_for_label["chaos_score"],
                rows_for_label["distance_score"],
                rows_for_label["surprise_score"],
                rows_for_label["tension_score"],
                rows_for_label["hover_row_data"],
            ], axis=-1) if len(rows_for_label) else np.empty((0, 5)),
            marker=dict(
                size=9,
                symbol=symbol_by_label.get(str(label), "diamond"),
                color=color_by_label.get(str(label), "blue"),
                opacity=0.9,
                line=dict(color="black", width=1),
            ),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "name: %{text}<br>"
                "normalized displacement: %{x:.3f}<br>"
                "normalized surprise: %{y:.3f}<br>"
                "normalized incongruity: %{z:.3f}<br>"
                "chaos score: %{customdata[0]:.3f}<br>"
                "distance score: %{customdata[1]:.3f}<br>"
                "surprise score: %{customdata[2]:.3f}<br>"
                "tension score: %{customdata[3]:.3f}<br>"
                "<br>%{customdata[4]}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Chaos Metric Space",
        width=950,
        height=760,
        scene=dict(
            xaxis=dict(title="Semantic displacement, normalized", range=[0, 1]),
            yaxis=dict(title="Contextual surprise, normalized", range=[0, 1]),
            zaxis=dict(title="Internal incongruity, normalized", range=[0, 1]),
            camera=dict(eye=dict(x=1.45, y=1.45, z=1.05)),
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


def write_chaos_metric_space_html(
    results: pd.DataFrame,
    output_html: str | Path,
    *,
    x_axis: str = "normalized_displacement",
    y_axis: str = "normalized_surprise",
    z_axis: str = "normalized_incongruity",
    detail_columns: Optional[Sequence[str]] = None,
    include_plotlyjs: str | bool = "cdn",
) -> Path:
    """Write an interactive HTML with a left dataframe-style detail panel.

    Hovering or clicking a point updates the table on the left. The 3D plot
    remains fully movable: drag rotates, scroll zooms, and pan controls work.
    """
    import plotly.io as pio

    plot_df = results.copy()
    if "status" in plot_df.columns:
        plot_df = plot_df[plot_df["status"].fillna("ok").eq("ok")].copy()

    if "normalized_displacement" not in plot_df.columns:
        plot_df["normalized_displacement"] = min_max_normalize_series(plot_df["distance_score"])
    if "normalized_surprise" not in plot_df.columns:
        plot_df["normalized_surprise"] = min_max_normalize_series(plot_df["surprise_score"])
    if "normalized_incongruity" not in plot_df.columns:
        plot_df["normalized_incongruity"] = min_max_normalize_series(plot_df["tension_score"])
    if "chaos_score" not in plot_df.columns:
        plot_df["chaos_score"] = plot_df[
            ["normalized_displacement", "normalized_surprise", "normalized_incongruity"]
        ].mean(axis=1)

    default_detail_columns = [
        "name",
        "idea",
        "standardized_idea",
        "cultural_observation",
        "problem_to_solve",
        "solution",
        "distance_score",
        "surprise_score",
        "tension_score",
        "basic_score",
        "universal_score",
        "status",
    ]
    detail_columns = [column for column in (detail_columns or default_detail_columns) if column in plot_df.columns]

    def row_payload(row: pd.Series) -> str:
        payload = {}
        for column in detail_columns:
            value = row.get(column)
            if value is None or pd.isna(value):
                value = ""
            elif isinstance(value, float):
                value = round(value, 6)
            else:
                value = clean_text(value)
            payload[column] = value
        return json.dumps(payload)

    plot_df["row_payload"] = [row_payload(row) for _, row in plot_df.iterrows()]
    fig = plot_chaos_metric_space(
        plot_df,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        hover_columns=["name", "idea", "distance_score", "surprise_score", "tension_score"],
    )

    for trace in fig.data:
        row_payloads = []
        trace_names = list(trace.text) if getattr(trace, "text", None) is not None else []
        for name in trace_names:
            matches = plot_df[plot_df["name"].astype(str).eq(str(name))] if "name" in plot_df.columns else plot_df.iloc[0:0]
            if len(matches):
                row_payloads.append(matches.iloc[0]["row_payload"])
            else:
                row_payloads.append("{}")
        trace.meta = row_payloads
        trace.hovertemplate = "<b>%{text}</b><extra></extra>"

    plot_html = pio.to_html(
        fig,
        include_plotlyjs=include_plotlyjs,
        full_html=False,
        div_id="chaos-plot",
    )

    initial_row = json.loads(plot_df.iloc[0]["row_payload"]) if len(plot_df) else {}
    initial_rows = "\n".join(
        f"<tr><th>{html.escape(str(key).replace('_', ' '))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in initial_row.items()
    )

    document = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chaos Metric Space</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #111;
      background: #f7f7f5;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(320px, 420px) 1fr;
      min-height: 100vh;
    }}
    .detail {{
      border-right: 1px solid #d8d8d4;
      background: #fff;
      padding: 18px;
      overflow: auto;
    }}
    .detail h1 {{
      margin: 0 0 12px;
      font-size: 18px;
      font-weight: 600;
    }}
    .hint {{
      margin: 0 0 16px;
      font-size: 13px;
      color: #555;
    }}
    #unlock-button {{
      appearance: none;
      border: 1px solid #d8d8d4;
      background: #f7f7f5;
      color: #111;
      padding: 7px 10px;
      margin: 0 0 12px;
      font: inherit;
      font-size: 13px;
      cursor: pointer;
    }}
    #unlock-button:hover {{
      background: #ededeb;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      line-height: 1.35;
    }}
    th, td {{
      border-top: 1px solid #e6e6e2;
      padding: 8px 0;
      vertical-align: top;
    }}
    th {{
      width: 34%;
      padding-right: 12px;
      text-align: left;
      color: #555;
      font-weight: 600;
      text-transform: capitalize;
    }}
    td {{
      overflow-wrap: anywhere;
    }}
    .plot {{
      min-width: 0;
      background: #f7f7f5;
    }}
    #chaos-plot {{
      width: 100%;
      height: 100vh;
    }}
    @media (max-width: 820px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .detail {{
        border-right: 0;
        border-bottom: 1px solid #d8d8d4;
        max-height: 42vh;
      }}
      #chaos-plot {{
        height: 70vh;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="detail">
      <h1 id="detail-title">Selected row</h1>
      <p class="hint" id="detail-hint">Hover a point to preview. Click a point to lock it.</p>
      <button id="unlock-button" type="button">Unlock selection</button>
      <table id="detail-table">
        <tbody>
          {initial_rows}
        </tbody>
      </table>
    </aside>
    <main class="plot">
      {plot_html}
    </main>
  </div>
  <script>
    const tableBody = document.querySelector("#detail-table tbody");
    const title = document.getElementById("detail-title");
    const hint = document.getElementById("detail-hint");
    const unlockButton = document.getElementById("unlock-button");
    let locked = false;
    function escapeHtml(value) {{
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}
    function renderDetails(payloadText, lockSelection = false) {{
      let payload = {{}};
      try {{
        payload = JSON.parse(payloadText || "{{}}");
      }} catch (error) {{
        payload = {{}};
      }}
      locked = lockSelection || locked;
      title.textContent = payload.name || "Selected row";
      hint.textContent = locked
        ? "Selection locked. Click another point to replace it, or unlock."
        : "Hover a point to preview. Click a point to lock it.";
      tableBody.innerHTML = Object.entries(payload)
        .filter(([, value]) => value !== null && value !== undefined && String(value).length > 0)
        .map(([key, value]) => `<tr><th>${{escapeHtml(key.replaceAll("_", " "))}}</th><td>${{escapeHtml(value)}}</td></tr>`)
        .join("");
    }}
    const plot = document.getElementById("chaos-plot");
    function getPayloadText(eventData) {{
      if (!eventData || !eventData.points || !eventData.points.length) return "{{}}";
      const point = eventData.points[0];
      return point.data.meta && point.data.meta[point.pointNumber];
    }}
    plot.on("plotly_hover", function(eventData) {{
      if (locked) return;
      renderDetails(getPayloadText(eventData), false);
    }});
    plot.on("plotly_click", function(eventData) {{
      locked = false;
      renderDetails(getPayloadText(eventData), true);
    }});
    unlockButton.addEventListener("click", function() {{
      locked = false;
      hint.textContent = "Hover a point to preview. Click a point to lock it.";
    }});
  </script>
</body>
</html>
"""

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path


def run_full_scoring_pipeline(
    ideas: pd.DataFrame,
    baseline: Dict[str, str],
    *,
    model_name: str = CAUSAL_LM_MODEL,
) -> pd.DataFrame:
    """Run Steps 2, 3, and 4 on a DataFrame of existing ideas."""
    scored = add_semantic_displacement_scores(ideas, baseline)
    scored = add_contextual_surprise_scores(scored, model_name=model_name)
    scored = add_relationship_incongruity_scores(scored)
    return scored


def load_saved_basicness_model(path: str | Path = "basicness_detector_model.pkl") -> Dict[str, object]:
    """Load the version-stable Basicness Detector artifact.

    The artifact stores plain logistic-regression parameters rather than a
    pickled sklearn estimator, so it can be used across sklearn versions.
    """
    model_path = Path(path)
    if not model_path.exists() and not model_path.is_absolute():
        candidates = [
            Path.cwd() / model_path,
            Path(__file__).resolve().parent / model_path,
            Path(__file__).resolve().parent.parent / model_path,
        ]
        model_path = next((candidate for candidate in candidates if candidate.exists()), model_path)

    with model_path.open("rb") as f:
        return pickle.load(f)
