"""
DLP False Positive Analysis

Pipeline:
  1. Load JSON logs for a tenant
  2. Isolate messages WITH detections (response_jsonl non-empty)
  3. Group detections by category AND detection_type (pattern vs contextual)
  4. Pre-score each group with heuristics
  5. For each category: sample representative messages, build entity-aware truncated payloads
  6. Send compact payloads to Azure OpenAI to estimate FP rate & identify patterns
  7. Output structured FP report with TP/FP counts and detection-type-gated recommendations

Usage:
  pip install numpy requests python-dotenv
  python dlp_fp_analysis.py --input logs.json --output fp_report.json [options]

Example:
  python dlp_fp_analysis.py --input request-logs-tenant-abc.json --max-categories 15 --samples-per-category 8 --debug
"""

import json
import os
import re
import time
import argparse
import logging
from collections import defaultdict
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from dotenv import load_dotenv

import numpy as np
import requests

load_dotenv()



# JSON helpers


def _json_default(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")



# Configuration


AZURE_OPENAI_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

HEADERS = {
    "Content-Type": "application/json",
    "api-key": AZURE_OPENAI_API_KEY,
}

DEFAULT_SAMPLES_PER_CATEGORY  = 8
DEFAULT_MIN_OCCURRENCES       = 2

PATTERN_DETECTION_FP_PRIOR    = 0.6
CONTEXTUAL_DETECTION_FP_PRIOR = 0.2

# Chars of surrounding context to keep on each side of a detected entity
ENTITY_CONTEXT_WINDOW = 300
# Hard cap on total input_text chars sent to LLM per example
MAX_SNIPPET_LENGTH    = 1200



# Logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)



# Rate limiter


class RateLimiter:
    def __init__(self, calls: int = 8, period: float = 60.0):
        self.calls  = calls
        self.period = period
        self._timestamps: List[float] = []

    def acquire(self):
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < self.period]
        if len(self._timestamps) >= self.calls:
            sleep_for = self.period - (now - self._timestamps[0])
            if sleep_for > 0:
                log.info(f"Rate limit: sleeping {sleep_for:.1f}s")
                time.sleep(sleep_for)
        self._timestamps.append(time.time())


_rate_limiter = RateLimiter()



# Azure OpenAI wrapper


def extract_json_from_response(raw_text: str) -> Optional[str]:
    raw_text = raw_text.strip()
    start = raw_text.find('{')
    end   = raw_text.rfind('}')
    if start != -1 and end != -1 and start < end:
        candidate = raw_text[start:end+1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`").strip()
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        return None


def azure_chat(prompt: str, temperature: float = 0.1, max_tokens: int = 3000) -> str:
    _rate_limiter.acquire()
    url = (
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/"
        f"{AZURE_OPENAI_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a Data Loss Prevention (DLP) expert and ML engineer. "
                    "Your job is to evaluate whether automated DLP detections on user prompts "
                    "are genuine (true positives) or incorrect (false positives). "
                    "Return ONLY valid JSON. No markdown, no comments, no explanations. "
                    "The JSON must follow the exact schema provided in the user message."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(url, headers=HEADERS, json=payload, timeout=90)
            r.raise_for_status()
            resp = r.json()

            choice        = resp["choices"][0]
            finish_reason = choice.get("finish_reason", "")
            raw_content   = choice.get("message", {}).get("content")

            log.debug(
                f"Raw LLM response (attempt {attempt}, finish_reason={finish_reason!r}):\n{raw_content}"
            )

            if finish_reason == "length":
                log.warning(
                    f"Response cut off by max_tokens={max_tokens}. "
                    "JSON likely incomplete — reduce payload size or raise max_tokens."
                )

            if not raw_content:
                raise ValueError(f"Empty response content (finish_reason={finish_reason!r})")

            extracted = extract_json_from_response(raw_content)
            if extracted is None:
                log.warning(
                    f"No valid JSON in response (attempt {attempt}). "
                    f"Preview:\n{raw_content[:500]}"
                )
                raise ValueError("No valid JSON found in response.")

            return extracted

        except (requests.HTTPError, requests.Timeout) as e:
            log.error(f"Azure API request error (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES:
                raise
        except ValueError as e:
            log.error(f"Response parsing error (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES:
                raise

        backoff = 2 ** attempt
        log.info(f"Retrying in {backoff}s…")
        time.sleep(backoff)

    raise RuntimeError("azure_chat exhausted all retries")



# Step 1: Load JSON


def load_json_robust(path: str) -> List[Dict]:
    encodings_to_try = [
        ("utf-8",     "strict"),
        ("utf-8-sig", "strict"),
        ("utf-8",     "replace"),
        ("latin-1",   "strict"),
    ]
    content    = None
    last_error = None
    for enc, err_mode in encodings_to_try:
        try:
            with open(path, "r", encoding=enc, errors=err_mode) as f:
                content = f.read()
            if enc != "utf-8" or err_mode != "strict":
                log.info(f"Loaded JSON with encoding={enc} errors={err_mode}")
            break
        except UnicodeDecodeError as e:
            last_error = e
    if content is None:
        raise last_error

    try:
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")
        return data
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error: {e}. Attempting recovery…")
        return _recover_json_array(content, e)


def _recover_json_array(content: str, original_error: json.JSONDecodeError) -> List[Dict]:
    a_start = content.find("[")
    a_end   = content.rfind("]")
    if a_start == -1 or a_end == -1:
        raise original_error

    body    = content[a_start + 1 : a_end]
    records = []
    depth, in_str, escape_next = 0, False, False
    quote_char, obj_start, skipped = None, None, 0
    i = 0
    while i < len(body):
        ch = body[i]
        if escape_next:
            escape_next = False; i += 1; continue
        if ch == "\\" and in_str:
            escape_next = True; i += 1; continue
        if ch in ('"', "'") and not in_str:
            in_str = True; quote_char = ch; i += 1; continue
        if ch == quote_char and in_str:
            in_str = False; quote_char = None; i += 1; continue
        if in_str:
            i += 1; continue
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    obj = json.loads(body[obj_start : i + 1])
                    if isinstance(obj, dict):
                        records.append(obj)
                except json.JSONDecodeError:
                    skipped += 1
                obj_start = None
        i += 1
    log.info(f"Recovered {len(records)} objects (skipped {skipped} malformed)")
    return records


def load_detected_only(path: str) -> Tuple[List[Dict], List[Dict]]:
    records = load_json_robust(path)
    detected, undetected = [], []
    for r in records:
        if not r.get("input_text") or r.get("errors"):
            continue
        (detected if r.get("response_jsonl") else undetected).append(r)
    log.info(
        f"Loaded {len(records)} records → "
        f"{len(detected)} detected (FP candidates), {len(undetected)} undetected"
    )
    return detected, undetected



# Step 2: Entity-aware input truncation


def _build_entity_aware_snippet(input_text: str, sensitive_entities: List[str]) -> str:
    if not input_text:
        return ""
    if not sensitive_entities:
        return _head_tail_split(input_text, MAX_SNIPPET_LENGTH)

    text_lower = input_text.lower()
    windows: List[Tuple[int, int]] = []

    for entity in sensitive_entities:
        if not entity:
            continue
        entity_lower = entity.lower().strip()
        pos = 0
        while True:
            idx = text_lower.find(entity_lower, pos)
            if idx == -1:
                break
            win_s = max(0, idx - ENTITY_CONTEXT_WINDOW)
            win_e = min(len(input_text), idx + len(entity) + ENTITY_CONTEXT_WINDOW)
            windows.append((win_s, win_e))
            pos = idx + 1

    if not windows:
        log.debug("Entities not found in input_text; falling back to head/tail split")
        return _head_tail_split(input_text, MAX_SNIPPET_LENGTH)

    # Merge overlapping windows
    windows.sort()
    merged: List[List[int]] = []
    for ws, we in windows:
        if merged and ws <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], we)
        else:
            merged.append([ws, we])

    # Assemble snippets, capped at MAX_SNIPPET_LENGTH total
    snippets   = []
    total_used = 0
    for ws, we in merged:
        prefix = "…" if ws > 0 else ""
        suffix = "…" if we < len(input_text) else ""
        chunk  = prefix + input_text[ws:we] + suffix
        remaining = MAX_SNIPPET_LENGTH - total_used
        if len(chunk) > remaining:
            if remaining > 60:
                snippets.append(chunk[:remaining] + "…")
            break
        snippets.append(chunk)
        total_used += len(chunk)

    return "\n---\n".join(snippets)


def _head_tail_split(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    half = max_len // 2 - 15
    return text[:half] + "\n…[middle truncated]…\n" + text[-half:]



# Step 3: Group detections by category


def _normalize_detection_type(detection: Dict) -> str:
    """
    Normalize to 'pattern', 'contextual', 'edm', or 'unknown'.
    'contextual' covers: contextual, semantic, model, ml.
    """
    raw = (detection.get("detection_type") or "").lower().strip()
    if raw == "pattern":
        return "pattern"
    if raw in ("contextual", "semantic", "model", "ml"):
        return "contextual"
    if raw == "edm":
        return "edm"
    return "unknown"


def group_by_category(detected_records: List[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for record in detected_records:
        for detection in record.get("response_jsonl", []):
            category = detection.get("name") or detection.get("code_name") or "Unknown"
            det_type = _normalize_detection_type(detection)
            fp_prior = (
                PATTERN_DETECTION_FP_PRIOR if det_type == "pattern"
                else CONTEXTUAL_DETECTION_FP_PRIOR
            )
            groups[category].append({
                "record":         record,
                "detection":      detection,
                "detection_type": det_type,
                "fp_prior":       fp_prior,
            })

    for cat, items in groups.items():
        p = sum(1 for i in items if i["detection_type"] == "pattern")
        c = sum(1 for i in items if i["detection_type"] == "contextual")
        log.info(f"Category '{cat}': {len(items)} detections (pattern={p}, contextual={c})")

    return dict(groups)


def _dominant_detection_type(items: List[Dict]) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for item in items:
        counts[item["detection_type"]] += 1
    return max(counts, key=lambda k: counts[k])



# Step 4: Heuristic pre-scoring and sampling


def _heuristic_fp_score(item: Dict) -> float:
    score     = item["fp_prior"]
    detection = item["detection"]
    record    = item["record"]

    for entity, subcat in detection.get("entity_texts_with_subcategories", {}).items():
        # Email flagged as UPI ID — known bad regex
        if re.match(r"[^@]+@[^@]+", entity) and "upi" in subcat.lower():
            score = min(score + 0.35, 1.0)
        # Email without proper TLD flagged as financial
        if re.match(r"^[\w.-]+@[\w-]+$", entity) and not re.search(r"\.(com|org|net|io|ai)$", entity):
            score = min(score + 0.2, 1.0)

    if len((record.get("input_text") or "").split()) < 20:
        score = min(score + 0.15, 1.0)

    entities = detection.get("sensitive_entities", [])
    if len(entities) > 3 and len(set(entities)) <= 2:
        # Same entity repeated many times → likely a forwarded thread, not new disclosure
        score = min(score + 0.15, 1.0)

    return round(score, 3)


def sample_for_category(
    items: List[Dict],
    n: int = DEFAULT_SAMPLES_PER_CATEGORY,
) -> Tuple[List[Dict], float]:
    for item in items:
        item["heuristic_fp_score"] = _heuristic_fp_score(item)
    sorted_items = sorted(items, key=lambda x: -x["heuristic_fp_score"])
    mean_prior   = float(np.mean([i["heuristic_fp_score"] for i in items]))
    return sorted_items[:n], mean_prior



# Step 5: Build LLM payload per category


def build_category_payload(
    category: str,
    sampled_items: List[Dict],
    total_count: int,
    mean_prior_score: float,
    dominant_det_type: str,
) -> Dict:
    """
    Build a compact, LLM-ready payload.

    Notable design decisions:
    - input_text uses entity-aware windowed truncation to preserve entity context
    - The full response_jsonl detection block is passed (not just extracted fields)
    - sensitive_entities are labelled as 'positive_examples' so the LLM understands
      they are what the system claimed was sensitive — not ground truth
    - dominant_detection_type is surfaced so prompt gating of recommendations works
    """
    examples = []
    for item in sampled_items:
        record    = item["record"]
        detection = item["detection"]

        raw_input          = record.get("input_text") or ""
        sensitive_entities = detection.get("sensitive_entities", [])

        input_snippet = _build_entity_aware_snippet(raw_input, sensitive_entities)

        examples.append({
            "input_text":          input_snippet,
            "input_was_truncated": len(raw_input) > MAX_SNIPPET_LENGTH,
            "full_detection": {
                # Pass the complete detection block — entity type, subcategory,
                # reason, detection_type, action type — everything the LLM needs
                "id":             detection.get("id"),
                "name":           detection.get("name"),
                "type":           detection.get("type"),
                "detection_type": detection.get("detection_type"),
                "reason":         detection.get("reason"),
                # Labelled explicitly: these are what the system flagged as sensitive
                # They are positive examples for the detection — NOT verified ground truth
                "sensitive_entities_positive_examples": sensitive_entities,
                "entity_texts_with_subcategories":      detection.get("entity_texts_with_subcategories", {}),
            },
            "heuristic_fp_score": item["heuristic_fp_score"],
        })

    return {
        "detection_category":      category,
        "dominant_detection_type": dominant_det_type,
        "total_detections":        total_count,
        "samples_shown":           len(examples),
        "mean_heuristic_fp_prior": round(mean_prior_score, 3),
        "examples":                examples,
    }



# Step 6: LLM Analysis


FP_ANALYSIS_PROMPT = """\
You are evaluating whether a DLP (Data Loss Prevention) detection category is producing false positives.

Each example below includes:
  - input_text: the user's prompt (may be windowed around detected entities if the original was long;
    "---" separators indicate window boundaries within the same message)
  - full_detection: the complete detection result, including:
      * detection_type: how the detection fired
          "pattern"    = regex/rule-based matching
          "contextual" = ML/semantic model
      * sensitive_entities_positive_examples: the EXACT strings the system flagged as sensitive.
        These are what the detector believed was sensitive — they are NOT verified ground truth.
        Your job is to judge whether each one is genuinely sensitive given the context.
      * entity_texts_with_subcategories: the sub-type each entity was classified as
        (e.g., "NATIONAL ID NUMBER", "UPI ID", "EMAIL_ADDRESS")
      * reason: the system's own explanation for the detection

HOW TO EVALUATE:
  1. Read input_text carefully — understand what the user is actually doing or asking.
  2. For each sensitive_entities_positive_example, ask:
       - Is this string actually a "{detection_category}" in this context?
       - Does the sub-type label make sense? (e.g., is an @email.com address really a UPI ID?)
       - Would a trained DLP analyst flag this as a genuine policy violation?
  3. Consider domain context: healthcare text may contain IDs that look like numbers;
     finance text may contain codes; IT text may contain hashes or tokens.
  4. A detection is a TRUE POSITIVE if the entity is genuinely sensitive in this context.
     A detection is a FALSE POSITIVE if the entity matched a pattern but is not actually sensitive.

Detection category data:
{payload_json}

─────────────────────────────────────────────
TUNING RECOMMENDATION RULES (strictly enforced):
  dominant_detection_type for this category = "{dominant_detection_type}"

  If dominant_detection_type is "pattern":
    → ONLY use: "disable", "tighten_regex", or "no_action"
    → NEVER use "add_allowlist" or "reduce_scope"
    → tighten_regex means: the underlying regex pattern needs to be narrowed or made more specific

  If dominant_detection_type is "contextual":
    → ONLY use: "disable", "add_allowlist", "reduce_scope", or "no_action"
    → NEVER use "tighten_regex"
    → add_allowlist means: exclude known-safe patterns/domains from triggering this model
    → reduce_scope means: narrow the detection to specific data types or contexts

  Shared options:
    → disable:   FPs are so overwhelming that the category provides little genuine signal
    → no_action: FP rate is within acceptable range; category is working correctly
─────────────────────────────────────────────

Respond with ONLY a JSON object in this exact format. Do not include any additional text, markdown, or explanations.
The JSON must adhere strictly to the schema below:
{{
  "detection_category": "<category name>",
  "dominant_detection_type": "{dominant_detection_type}",
  "estimated_fp_rate": <float 0.0–1.0>,
  "fp_rate_confidence": <float 0.0–1.0>,
  "estimated_tp_count_in_sample": <int: how many shown samples you judged TP>,
  "estimated_fp_count_in_sample": <int: how many shown samples you judged FP>,
  "fp_pattern_summary": "<1-2 sentences: what is causing the FPs, or null if none>",
  "tuning_recommendation": "disable" | "tighten_regex" | "add_allowlist" | "reduce_scope" | "no_action",
  "tuning_rationale": "<1-2 sentences explaining the recommendation>",
  "per_example_verdicts": [
    {{
      "input_snippet": "<first 100 chars of input_text for this example>",
      "verdict": "TP" | "FP" | "UNCERTAIN",
      "reason": "<brief: name the specific entity and why it is or isn't genuinely sensitive>"
    }}
  ],
  "suggested_allowlist_patterns": ["<pattern1>", "<pattern2>"],
  "risk_of_disabling": "low" | "medium" | "high",
  "overall_assessment": "high_fp" | "moderate_fp" | "low_fp" | "no_fp"
}}

overall_assessment thresholds:
  high_fp:     >60% FP rate  — misconfigured or too broad
  moderate_fp: 30–60% FP rate — tuning needed
  low_fp:      10–30% FP rate — minor tuning may help
  no_fp:       <10% FP rate  — working correctly

suggested_allowlist_patterns: regex or literal strings to add as allowlist exclusions.
  Example: ["@veg\\.com$", "\\b[A-Z]{{1}}\\d{{7}}\\b"]
  Return [] if not applicable.
"""


def analyze_category(payload: Dict) -> Optional[Dict]:
    payload_json  = json.dumps(payload, indent=2, default=_json_default)
    dominant_type = payload.get("dominant_detection_type", "unknown")
    prompt = FP_ANALYSIS_PROMPT.format(
        payload_json=payload_json,
        dominant_detection_type=dominant_type,
        detection_category=payload.get("detection_category", ""),
    )

    try:
        raw_json = azure_chat(prompt, temperature=0.1, max_tokens=3000)
        return json.loads(raw_json)
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error for category '{payload['detection_category']}': {e}")
        return None
    except (requests.HTTPError, ValueError, RuntimeError) as e:
        log.error(f"Failed to analyze category '{payload['detection_category']}': {e}")
        return None


def analyze_all_categories(
    category_groups: Dict[str, List[Dict]],
    samples_per_category: int,
    min_occurrences: int,
    max_categories: Optional[int],
) -> List[Dict]:
    category_payloads = []
    for category, items in category_groups.items():
        if len(items) < min_occurrences:
            log.info(f"Skipping '{category}' — only {len(items)} occurrence(s) (min={min_occurrences})")
            continue
        sampled, mean_prior = sample_for_category(items, n=samples_per_category)
        dominant            = _dominant_detection_type(items)
        payload             = build_category_payload(
            category, sampled, len(items), mean_prior, dominant
        )
        category_payloads.append((category, payload, mean_prior))

    # Prioritize highest heuristic FP prior (pattern-heavy categories first)
    category_payloads.sort(key=lambda x: -x[2])

    if max_categories is not None:
        category_payloads = category_payloads[:max_categories]
        log.info(f"Analyzing top {len(category_payloads)} categories (--max-categories={max_categories})")

    results = []
    total   = len(category_payloads)
    for i, (category, payload, mean_prior) in enumerate(category_payloads):
        log.info(
            f"Analyzing '{category}' ({i+1}/{total}) "
            f"[{payload['dominant_detection_type']}] — heuristic FP prior: {mean_prior:.2f}…"
        )
        result = analyze_category(payload)
        if result:
            total_n = len(category_groups[category])
            fp_rate = result.get("estimated_fp_rate") or 0.0

            result["total_detections_in_period"] = total_n
            result["mean_heuristic_fp_prior"]    = round(mean_prior, 3)
            result["estimated_fp_count_total"]   = round(fp_rate * total_n)
            result["estimated_tp_count_total"]   = total_n - round(fp_rate * total_n)
            results.append(result)
        else:
            log.warning(f"No valid response for category '{category}'")

    results.sort(key=lambda x: -x.get("estimated_fp_rate", 0))
    log.info(f"FP analysis complete: {len(results)} categories analyzed")
    return results



# Step 7: Build Report


ASSESSMENT_ORDER = {"high_fp": 0, "moderate_fp": 1, "low_fp": 2, "no_fp": 3}
RISK_COLORS      = {"high_fp": "🔴", "moderate_fp": "🟡", "low_fp": "🟢", "no_fp": "✅"}


def build_fp_report(
    tenant_id: str,
    all_results: List[Dict],
    detected_count: int,
    undetected_count: int,
    total_count: int,
) -> Dict:
    high_fp_cats     = [r for r in all_results if r.get("overall_assessment") == "high_fp"]
    moderate_fp_cats = [r for r in all_results if r.get("overall_assessment") == "moderate_fp"]

    sorted_for_action = sorted(
        all_results,
        key=lambda x: ASSESSMENT_ORDER.get(x.get("overall_assessment", "no_fp"), 4),
    )

    return {
        "tenant_id":    tenant_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_messages":             total_count,
            "messages_with_detection":    detected_count,
            "messages_without_detection": undetected_count,
            "categories_analyzed":        len(all_results),
            "high_fp_categories":         len(high_fp_cats),
            "moderate_fp_categories":     len(moderate_fp_cats),
        },
        "action_required": [
            {
                "rank":                    i + 1,
                "category":                r["detection_category"],
                "detection_type":          r.get("dominant_detection_type"),
                "assessment":              r.get("overall_assessment"),
                "estimated_fp_rate":       r.get("estimated_fp_rate"),
                "fp_rate_confidence":      r.get("fp_rate_confidence"),
                # Counts across full period (estimated from FP rate × total)
                "total_detections":        r.get("total_detections_in_period"),
                "estimated_tp_total":      r.get("estimated_tp_count_total"),
                "estimated_fp_total":      r.get("estimated_fp_count_total"),
                # Sample-level verdicts from the LLM
                "sample_tp_count":         r.get("estimated_tp_count_in_sample"),
                "sample_fp_count":         r.get("estimated_fp_count_in_sample"),
                "fp_pattern_summary":      r.get("fp_pattern_summary"),
                "tuning_recommendation":   r.get("tuning_recommendation"),
                "tuning_rationale":        r.get("tuning_rationale"),
                "suggested_allowlist":     r.get("suggested_allowlist_patterns", []),
                "risk_of_disabling":       r.get("risk_of_disabling"),
                "mean_heuristic_fp_prior": r.get("mean_heuristic_fp_prior"),
            }
            for i, r in enumerate(sorted_for_action)
            if r.get("overall_assessment") in ("high_fp", "moderate_fp")
        ],
        "all_category_results": all_results,
    }


def print_summary(report: Dict):
    print("\n" + "="*74)
    print("  DLP FALSE POSITIVE ANALYSIS REPORT")
    print("="*74)
    s = report["summary"]
    print(f"  Tenant:               {report['tenant_id']}")
    print(f"  Total messages:       {s['total_messages']}")
    print(f"  With detections:      {s['messages_with_detection']}")
    print(f"  Categories analyzed:  {s['categories_analyzed']}")
    print(f"  High FP categories:   {s['high_fp_categories']}")
    print(f"  Moderate FP:          {s['moderate_fp_categories']}")
    print()

    if not report["action_required"]:
        print("  ✅  All categories appear to be within acceptable FP rates.")
    else:
        print("─── Categories Requiring Action ─────────────────────────────────────")
        for item in report["action_required"]:
            icon   = RISK_COLORS.get(item["assessment"], "⚪")
            fp_pct = f"{item['estimated_fp_rate']:.0%}" if item["estimated_fp_rate"] is not None else "N/A"
            conf   = f"{item['fp_rate_confidence']:.0%}" if item["fp_rate_confidence"] is not None else "N/A"
            total  = item["total_detections"] or 0
            print(f"\n  #{item['rank']}  {icon}  {item['category']}")
            print(f"       Detection type:   {item['detection_type']}")
            print(f"       FP Rate:          ~{fp_pct}  (model confidence: {conf})")
            print(f"       Period totals:    {total} detections  →  "
                  f"~{item.get('estimated_tp_total','?')} TP  /  ~{item.get('estimated_fp_total','?')} FP")
            print(f"       Sample verdicts:  {item.get('sample_tp_count','?')} TP  /  "
                  f"{item.get('sample_fp_count','?')} FP  (of {DEFAULT_SAMPLES_PER_CATEGORY} shown to LLM)")
            print(f"       Action:           {item['tuning_recommendation'].replace('_',' ').upper()}")
            print(f"       FP pattern:       {item['fp_pattern_summary']}")
            print(f"       Rationale:        {item['tuning_rationale']}")
            if item["suggested_allowlist"]:
                print(f"       Allowlist:        {item['suggested_allowlist'][:3]}")
            print(f"       Risk if disabled: {item['risk_of_disabling'].upper()}")

    print("\n" + "─── All Categories at a Glance ───────────────────────────────────────")
    for r in report["all_category_results"]:
        icon     = RISK_COLORS.get(r.get("overall_assessment"), "⚪")
        fp_pct   = f"{r.get('estimated_fp_rate', 0):.0%}"
        det_type = r.get("dominant_detection_type", "?")
        tp_t     = r.get("estimated_tp_count_total", "?")
        fp_t     = r.get("estimated_fp_count_total", "?")
        total_n  = r.get("total_detections_in_period", "?")
        rec      = r.get("tuning_recommendation", "?")
        print(
            f"  {icon}  {r['detection_category']:<45} "
            f"[{det_type:<10}]  FP~{fp_pct:<5} "
            f"TP={tp_t}/{total_n}  [{rec}]"
        )
    print("\n" + "="*74 + "\n")



# Main


def main():
    parser = argparse.ArgumentParser(description="DLP False Positive Analysis")
    parser.add_argument("--input",   required=True, help="Path to JSON log file")
    parser.add_argument(
        "--output",
        default=f"dlp_fp_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--samples-per-category", type=int, default=DEFAULT_SAMPLES_PER_CATEGORY,
        help="Representative messages sent to LLM per detection category (default: 8)",
    )
    parser.add_argument(
        "--min-occurrences", type=int, default=DEFAULT_MIN_OCCURRENCES,
        help="Skip categories with fewer than N detections (default: 2)",
    )
    parser.add_argument(
        "--max-categories", type=int, default=None,
        help="Limit LLM calls to top N categories by FP prior (default: all)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG logging — prints raw LLM responses for diagnosis",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Load ─────────────────────────────────
    detected, undetected = load_detected_only(args.input)
    all_records = detected + undetected
    tenant_id   = all_records[0].get("tenant_id", "unknown") if all_records else "unknown"

    if not detected:
        log.info("No detected messages found. Nothing to analyze for false positives.")
        return

    # ── Group by category ────────────────────
    category_groups = group_by_category(detected)
    if not category_groups:
        log.info("No detection categories found in the data.")
        return

    # ── Skip LLM if no credentials ───────────
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        log.warning("Azure credentials not set — dumping raw payloads without LLM analysis.")
        log.warning("Set: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT")
        raw_payloads = []
        for category, items in category_groups.items():
            sampled, mean_prior = sample_for_category(items, n=args.samples_per_category)
            dominant            = _dominant_detection_type(items)
            raw_payloads.append(
                build_category_payload(category, sampled, len(items), mean_prior, dominant)
            )
        with open(args.output, "w") as f:
            json.dump(raw_payloads, f, indent=2, default=_json_default)
        log.info(f"Raw payloads saved to {args.output}")
        return

    # ── Analyze ──────────────────────────────
    all_results = analyze_all_categories(
        category_groups,
        samples_per_category=args.samples_per_category,
        min_occurrences=args.min_occurrences,
        max_categories=args.max_categories,
    )

    # ── Report ───────────────────────────────
    report = build_fp_report(
        tenant_id,
        all_results,
        detected_count=len(detected),
        undetected_count=len(undetected),
        total_count=len(all_records),
    )

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)

    log.info(f"FP report saved to {args.output}")
    print_summary(report)


if __name__ == "__main__":
    main()