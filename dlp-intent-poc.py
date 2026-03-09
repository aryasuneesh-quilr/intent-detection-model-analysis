"""
DLP Intent Suggestion POC (Enhanced)

Pipeline:
  1. Load JSON logs for a tenant
  2. Split into detected vs undetected messages
  3. Preprocess + deduplicate undetected messages (Jaccard or MinHash LSH)
  4. Embed with sentence-transformers (local, free)
  5. (Optional) Reduce dimensions with UMAP
  6. Cluster with DBSCAN or HDBSCAN
  7. Extract cluster representatives + TF-IDF keywords
  8. Send compact payloads to Azure OpenAI for DLP intent analysis
  9. Output structured suggestions

Usage:
  pip install sentence-transformers scikit-learn numpy requests umap-learn hdbscan datasketch
  python dlp_intent_poc.py --input logs.json --output suggestions.json [options]

Example:
  python dlp-intent-poc.py --input request-logs-tenant-abc.json --max-clusters 20 --use-umap --clustering-algorithm hdbscan --dedup-method minhash
"""

import json
import os
import re
import time
import argparse
import logging
from collections import defaultdict
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Set
from dotenv import load_dotenv

import numpy as np
import requests
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False

load_dotenv()



# JSON helpers (NumPy types aren't JSON-serializable by default)

def _json_default(obj):
    """Convert NumPy types to native Python for json.dump/dumps."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")



# Configuration (defaults)

AZURE_OPENAI_ENDPOINT  = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY   = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")  # your deployment name

HEADERS = {
    "Content-Type": "application/json",
    "api-key": AZURE_OPENAI_API_KEY,
}

# Clustering defaults – can be overridden by command line
DEFAULT_DBSCAN_EPS         = 0.35
DEFAULT_DBSCAN_MIN_SAMPLES = 3
DEFAULT_MIN_TOKEN_LENGTH   = 15     # discard messages shorter than this (word count)
DEFAULT_REPRESENTATIVES_N  = 5      # messages sent to LLM per cluster
DEFAULT_KEYWORDS_N         = 10     # TF-IDF keywords shown per cluster

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # local, fast, free



# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)



# Rate limiter (simple token bucket)

class RateLimiter:
    """Simple rate limiter: max `calls` per `period` seconds."""
    def __init__(self, calls: int = 10, period: float = 60.0):
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


_rate_limiter = RateLimiter(calls=8, period=60)



# Azure OpenAI wrapper with robust JSON extraction

def extract_json_from_response(raw_text: str) -> Optional[str]:
    """
    Extract a JSON object from the raw response by finding the first '{' and last '}'.
    Falls back to removing markdown fences if that fails.
    """
    # Remove leading/trailing whitespace
    raw_text = raw_text.strip()
    # Find the first '{' and the last '}'
    start = raw_text.find('{')
    end = raw_text.rfind('}')
    if start != -1 and end != -1 and start < end:
        candidate = raw_text[start:end+1]
        # Quick validation: try to parse
        try:
            json.loads(candidate)
            log.debug("Extracted JSON using brace matching.")
            return candidate
        except json.JSONDecodeError:
            pass
    # Fallback: remove markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`").strip()
    # Try to see if that yields valid JSON
    try:
        json.loads(cleaned)
        log.debug("Extracted JSON after removing markdown fences.")
        return cleaned
    except json.JSONDecodeError:
        log.warning("Could not extract valid JSON from response.")
        return None


def azure_chat(prompt: str, temperature: float = 0.2, max_tokens: int = 2500) -> str:
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
                    "You are a Data Loss Prevention (DLP) expert. "
                    "Your job is to analyze clusters of user prompts sent to an AI chatbot "
                    "and identify if they contain or suggest sensitive data patterns "
                    "that the customer's DLP system has not yet configured detection for. "
                    "Return ONLY valid JSON. No markdown, no comments, no explanations. "
                    "The JSON must follow the exact schema provided in the user message."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(url, headers=HEADERS, json=payload, timeout=90)
        r.raise_for_status()
        resp = r.json()
        choice = resp["choices"][0]
        finish_reason = choice.get("finish_reason", "")
        message = choice.get("message", {})
        if "content" not in message or message["content"] is None:
            raise ValueError(f"No content in response (finish_reason={finish_reason!r})")
        raw = message["content"]
        extracted = extract_json_from_response(raw)
        if extracted is None:
            raise ValueError("No valid JSON found in response.")
        return extracted
    except requests.HTTPError as e:
        log.error(f"Azure API HTTP error: {e}")
        raise
    except Exception as e:
        log.error(f"Azure API call failed: {e}")
        raise



# Step 1: Load & Split

def load_json_robust(path: str) -> List[Dict]:
    """
    Load JSON with robust encoding and optional error recovery.
    Tries UTF-8 first, then fallback encodings. On JSONDecodeError,
    attempts to recover array-of-objects by extracting valid objects.
    """
    encodings_to_try = [
        ("utf-8", "strict"),
        ("utf-8-sig", "strict"),  # handle BOM
        ("utf-8", "replace"),     # replace invalid bytes
        ("latin-1", "strict"),    # rarely fails, may mangle some chars
    ]
    content = None
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
            continue
    if content is None:
        raise last_error

    try:
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array of records, got {type(data).__name__}")
        return data
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error: {e}. Attempting recovery for array-of-objects...")
        return _recover_json_array_of_objects(content, e)


def _recover_json_array_of_objects(content: str, parse_error: json.JSONDecodeError) -> List[Dict]:
    """
    Extract valid objects from malformed JSON array like [{...}, {...}].
    Uses brace matching to find object boundaries; skips malformed entries.
    """
    array_start = content.find("[")
    array_end = content.rfind("]")
    if array_start == -1 or array_end == -1:
        raise parse_error

    array_content = content[array_start + 1 : array_end]
    records = []
    depth = 0
    obj_start = None
    in_string = False
    escape_next = False
    quote_char = None
    skipped = 0

    i = 0
    while i < len(array_content):
        char = array_content[i]
        if escape_next:
            escape_next = False
            i += 1
            continue
        if char == "\\" and in_string:
            escape_next = True
            i += 1
            continue
        if char in ('"', "'") and not in_string:
            in_string = True
            quote_char = char
            i += 1
            continue
        if char == quote_char and in_string:
            in_string = False
            quote_char = None
            i += 1
            continue
        if in_string:
            i += 1
            continue

        if char == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                chunk = array_content[obj_start : i + 1]
                try:
                    obj = json.loads(chunk)
                    if isinstance(obj, dict):
                        records.append(obj)
                except json.JSONDecodeError:
                    skipped += 1
                obj_start = None
        i += 1

    log.info(f"Recovered {len(records)} objects (skipped {skipped} malformed)")
    return records


def load_and_split(path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load JSON log file and split into detected vs undetected messages."""
    records = load_json_robust(path)

    detected   = []
    undetected = []

    for r in records:
        # Skip records with errors or missing input
        if not r.get("input_text") or r.get("errors"):
            continue

        has_detection = bool(r.get("response_jsonl"))

        if has_detection:
            detected.append(r)
        else:
            undetected.append(r)

    log.info(f"Loaded {len(records)} records → {len(detected)} detected, {len(undetected)} undetected")
    return detected, undetected



# Step 2: Preprocess undetected messages (with pluggable deduplication)

def _word_shingles(text: str, k: int = 3) -> Set[str]:
    """k-word shingles for near-duplicate detection."""
    words = text.lower().split()
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}


def _jaccard(a: Set, b: Set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _minhash_lsh_deduplicate(texts: List[str], threshold: float = 0.8, num_perm: int = 128) -> List[int]:
    """
    Use MinHash LSH to find unique texts (keeping first occurrence of each near-duplicate group).
    Returns indices of texts to keep.
    """
    if not DATASKETCH_AVAILABLE:
        raise ImportError("datasketch is not installed. Please install it: pip install datasketch")
    # Create LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    # Store MinHash objects for each text
    minhashes = []
    for text in texts:
        m = MinHash(num_perm=num_perm)
        for word in text.lower().split():
            m.update(word.encode('utf8'))
        minhashes.append(m)
    # Determine unique indices
    unique_indices = []
    for i, m in enumerate(minhashes):
        # Check if a similar MinHash already exists in LSH
        result = lsh.query(m)
        if not result:
            lsh.insert(f"doc_{i}", m)
            unique_indices.append(i)
    log.info(f"MinHash LSH: kept {len(unique_indices)} / {len(texts)} unique messages (threshold={threshold})")
    return unique_indices


def _normalize_text(text: str) -> str:
    """Basic normalization: collapse whitespace, lowercase."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(records: List[Dict], dedup_method: str = 'jaccard', minhash_threshold: float = 0.8) -> List[Dict]:
    """
    Filter and deduplicate undetected records.
    Returns cleaned records with a `clean_text` field added.
    """
    cleaned = []

    # First, apply length filter and basic normalization
    normalized_records = []
    for r in records:
        text = _normalize_text(r["input_text"])
        if len(text.split()) < DEFAULT_MIN_TOKEN_LENGTH:
            continue
        r = dict(r)  # copy
        r["clean_text"] = text
        normalized_records.append(r)

    if not normalized_records:
        return []

    # Deduplication
    texts = [r["clean_text"] for r in normalized_records]

    if dedup_method == 'jaccard':
        # Original Jaccard with shingles
        seen_shingles: List[Set] = []
        unique_indices = []
        for idx, text in enumerate(texts):
            shingles = _word_shingles(text)
            is_dup = any(
                _jaccard(shingles, s) >= DEDUP_THRESHOLD  # use global DEDUP_THRESHOLD from args
                for s in seen_shingles
            )
            if not is_dup:
                seen_shingles.append(shingles)
                unique_indices.append(idx)
        log.info(f"Jaccard dedup: kept {len(unique_indices)} / {len(texts)} unique messages")
    elif dedup_method == 'minhash':
        # MinHash LSH
        try:
            unique_indices = _minhash_lsh_deduplicate(texts, threshold=minhash_threshold)
        except ImportError as e:
            log.error(f"MinHash LSH not available: {e}. Falling back to Jaccard.")
            # Fallback to jaccard
            dedup_method = 'jaccard'
            unique_indices = list(range(len(texts)))  # keep all for now, but we need to re-run Jaccard properly
            # For simplicity, we'll just use all indices; but better to fallback properly.
            # We'll do the Jaccard approach here.
            seen_shingles = []
            unique_indices = []
            for idx, text in enumerate(texts):
                shingles = _word_shingles(text)
                is_dup = any(
                    _jaccard(shingles, s) >= DEDUP_THRESHOLD
                    for s in seen_shingles
                )
                if not is_dup:
                    seen_shingles.append(shingles)
                    unique_indices.append(idx)
            log.info(f"Fallback Jaccard dedup: kept {len(unique_indices)} / {len(texts)} unique messages")
    else:
        raise ValueError(f"Unknown dedup method: {dedup_method}")

    cleaned = [normalized_records[i] for i in unique_indices]
    log.info(f"After preprocessing: {len(cleaned)} unique messages (from {len(records)})")
    return cleaned



# Step 3: Embed

def embed(records: List[Dict]) -> np.ndarray:
    """Generate sentence embeddings for all clean_text fields."""
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [r["clean_text"] for r in records]
    log.info(f"Embedding {len(texts)} messages…")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    log.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings



# Step 4: Dimensionality Reduction (UMAP)

def reduce_dimensions(embeddings: np.ndarray, n_components: int = 50,
                      n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """Apply UMAP for dimensionality reduction."""
    if not UMAP_AVAILABLE:
        log.warning("UMAP not installed. Skipping dimensionality reduction.")
        return embeddings
    log.info(f"Reducing dimensions from {embeddings.shape[1]} to {n_components} with UMAP...")
    reducer = umap.UMAP(n_components=n_components, random_state=42,
                        n_neighbors=n_neighbors, min_dist=min_dist)
    embeddings_reduced = reducer.fit_transform(embeddings)
    log.info(f"UMAP reduction complete. New shape: {embeddings_reduced.shape}")
    return embeddings_reduced



# Step 5: Clustering (DBSCAN or HDBSCAN)

def cluster(embeddings: np.ndarray,
            algorithm: str = 'dbscan',
            eps: float = DEFAULT_DBSCAN_EPS,
            min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES) -> np.ndarray:
    """
    Cluster embeddings using DBSCAN or HDBSCAN.
    Returns array of cluster labels (-1 = noise/outlier).
    """
    # Normalize for cosine distance (if not already)
    X = normalize(embeddings)   # L2 norm → cosine distance = euclidean distance on unit sphere

    if algorithm == 'dbscan':
        log.info(f"Running DBSCAN (eps={eps}, min_samples={min_samples})…")
        db = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="cosine",
            n_jobs=-1,
        ).fit(X)
        labels = db.labels_
    elif algorithm == 'hdbscan':
        if not HDBSCAN_AVAILABLE:
            log.error("HDBSCAN not installed. Falling back to DBSCAN.")
            return cluster(embeddings, algorithm='dbscan', eps=eps, min_samples=min_samples)
        log.info(f"Running HDBSCAN (min_cluster_size={min_samples})…")
        # HDBSCAN uses min_cluster_size similar to min_samples in DBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_samples,
                                    metric='euclidean',  # on normalized embeddings = cosine
                                    prediction_data=True)
        labels = clusterer.fit_predict(X)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    log.info(f"Clustering completed: {n_clusters} clusters, {n_noise} noise points")
    return labels



# Step 6: Extract cluster representatives + keywords

def get_tfidf_keywords(texts: List[str], top_n: int = DEFAULT_KEYWORDS_N) -> List[str]:
    """Extract top TF-IDF keywords from a list of texts."""
    if len(texts) < 2:
        # TF-IDF needs at least 2 docs; fall back to most common words
        words = " ".join(texts).lower().split()
        freq: Dict[str, int] = defaultdict(int)
        for w in words:
            if len(w) > 3:
                freq[w] += 1
        return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_n]]

    try:
        tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=300,
            ngram_range=(1, 2),
        )
        X = tfidf.fit_transform(texts)
        scores = np.asarray(X.mean(axis=0)).flatten()
        terms  = tfidf.get_feature_names_out()
        top    = sorted(zip(terms, scores), key=lambda x: -x[1])[:top_n]
        return [t[0] for t in top]
    except Exception as e:
        log.warning(f"TF-IDF failed: {e}")
        return []


def extract_representatives(
    cluster_records: List[Dict],
    cluster_embeddings: np.ndarray,
    n: int = DEFAULT_REPRESENTATIVES_N,
) -> List[str]:
    """Pick the n messages closest to the cluster centroid."""
    centroid  = cluster_embeddings.mean(axis=0, keepdims=True)
    distances = cosine_distances(centroid, cluster_embeddings).flatten()
    top_idx   = distances.argsort()[:n]
    return [cluster_records[i]["clean_text"] for i in top_idx]


def build_cluster_payloads(
    records: List[Dict],
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> List[Dict]:
    """
    Build a compact payload per cluster (and one for noise points).
    Each payload contains: cluster_id, size, keywords, representatives.
    """
    # Group by label
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)

    payloads = []

    for label, indices in sorted(groups.items()):
        cluster_records    = [records[i] for i in indices]
        cluster_embeddings = embeddings[indices]
        texts              = [r["clean_text"] for r in cluster_records]

        keywords       = get_tfidf_keywords(texts)
        representatives = extract_representatives(cluster_records, cluster_embeddings)

        # Date range for context
        timestamps = [r.get("created_at", "") for r in cluster_records]
        timestamps = sorted([t for t in timestamps if t])

        payload = {
            "cluster_id":     label if label != -1 else "noise",
            "message_count":  len(cluster_records),
            "date_range":     {
                "from": timestamps[0] if timestamps else None,
                "to":   timestamps[-1] if timestamps else None,
            },
            "top_keywords":           keywords,
            "representative_messages": representatives,
        }
        payloads.append(payload)
        log.info(
            f"Cluster {'noise' if label == -1 else label}: "
            f"{len(cluster_records)} messages, keywords: {keywords[:5]}"
        )

    return payloads



# Step 7: LLM Analysis (with improved prompt)

ANALYSIS_PROMPT = """\
You are analyzing a cluster of user prompts that were sent to an AI chatbot by employees of a company.
The company has a DLP (Data Loss Prevention) system, but NO custom detection models were triggered for these prompts.

Your job:
1. Determine if this cluster contains or suggests sensitive/confidential data patterns.
2. If yes, identify what type of DLP detection model the company should consider enabling.
3. Rate your confidence (0.0 - 1.0).

Cluster data:
{cluster_json}

Respond with ONLY a JSON object in this exact format. Do not include any additional text, markdown, or explanations.
The JSON must adhere strictly to the schema below:

{{
  "cluster_id": "<cluster_id>",
  "is_sensitive": true | false,
  "sensitivity_confidence": <0.0 - 1.0>,
  "detected_intent_category": "<category or null>",
  "intent_description": "<1-2 sentence description of what sensitive pattern you see, or null; don't mention words like "cluster" in the description. Start with "This intent is for ...">",
  "suggested_dlp_model": "<name of DLP model to suggest to the customer, or null>",
  "suggested_model_rationale": "<why this model would help, or null>",
  "example_triggering_phrases": ["<phrase1>", "<phrase2>"] (Mandatory: these are the exact phrases that triggered the detection),
  "example_counter_phrases": ["<phrase1>", "<phrase2>"] (Mandatory: these are the exact phrases that you can generate that counter the detection - think of confusing phrases that might get mistaken for the triggering phrase but it is a negative example for the detection model),
  "risk_level": "low" | "medium" | "high" | "critical"
}}

Intent category must be one of the following:
  - PII (names, addresses, personal identifiers)
  - PHI (health/medical information)
  - PFI (financial data, account numbers, costs, pricing)
  - CREDENTIALS (passwords, API keys, tokens, connection strings)
  - INTELLECTUAL_PROPERTY (proprietary processes, trade secrets, internal strategy)
  - LEGAL (contracts, litigation, compliance discussions)
  - HR_SENSITIVE (employee data, performance, salary, terminations)
  - INTERNAL_COMMS (internal email chains, org structure, personnel discussions)
  - OTHER_SENSITIVE
  - NOT_SENSITIVE

If the cluster is NOT sensitive, set `is_sensitive` to false, and all other fields (except cluster_id) to null.
If it is sensitive, choose the most appropriate category from the list; if none fit, use OTHER_SENSITIVE.
Risk level should reflect the potential impact of exposure.
"""


def analyze_cluster(payload: Dict) -> Optional[Dict]:
    """Send a single cluster payload to Azure OpenAI and parse the response."""
    cluster_json = json.dumps(payload, indent=2, default=_json_default)
    prompt = ANALYSIS_PROMPT.format(cluster_json=cluster_json)

    try:
        raw_json = azure_chat(prompt, temperature=0.1, max_tokens=1000)
        result = json.loads(raw_json)
        return result
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error for cluster {payload['cluster_id']}: {e}")
        return None
    except requests.HTTPError as e:
        log.error(f"Azure API error for cluster {payload['cluster_id']}: {e}")
        return None


def analyze_all_clusters(payloads: List[Dict]) -> List[Dict]:
    """Analyze all cluster payloads and return only sensitive ones."""
    results = []
    log.info(f"Sending {len(payloads)} clusters to Azure OpenAI for analysis…")

    for i, payload in enumerate(payloads):
        log.info(f"Analyzing cluster {payload['cluster_id']} ({i+1}/{len(payloads)})…")
        result = analyze_cluster(payload)

        if result:
            # Attach original cluster stats for context
            result["message_count"]  = payload["message_count"]
            result["top_keywords"]   = payload["top_keywords"]
            result["date_range"]     = payload["date_range"]
            results.append(result)
        else:
            log.warning(f"Skipping cluster {payload['cluster_id']} — no valid response")

    sensitive = [r for r in results if r.get("is_sensitive")]
    log.info(
        f"Analysis complete: {len(sensitive)}/{len(results)} clusters flagged as sensitive"
    )
    return results



# Step 8: Format Output Report

def build_report(
    tenant_id: str,
    all_results: List[Dict],
    detected_records: List[Dict],
    undetected_count: int,
    total_count: int,
) -> Dict:
    """Combine everything into a structured final report."""

    # Summarize already-detected intents (from response_jsonl)
    detected_intents: Dict[str, int] = defaultdict(int)
    for r in detected_records:
        for det in r.get("response_jsonl", []):
            detected_intents[det.get("name", "Unknown")] += 1

    sensitive_clusters = [r for r in all_results if r.get("is_sensitive")]
    sensitive_clusters.sort(key=lambda x: -x.get("sensitivity_confidence", 0))

    report = {
        "tenant_id":   tenant_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_messages":                   total_count,
            "messages_with_existing_detection": len(detected_records),
            "messages_without_detection":       undetected_count,
            "clusters_analyzed":                len(all_results),
            "sensitive_clusters_found":         len(sensitive_clusters),
        },
        "existing_detections": dict(detected_intents),
        "suggested_new_models": [
            {
                "rank":                     i + 1,
                "cluster_id":               c["cluster_id"],
                "risk_level":               c.get("risk_level"),
                "confidence":               c.get("sensitivity_confidence"),
                "intent_category":          c.get("detected_intent_category"),
                "intent_description":       c.get("intent_description"),
                "suggested_dlp_model":      c.get("suggested_dlp_model"),
                "rationale":                c.get("suggested_model_rationale"),
                "example_triggers":         c.get("example_triggering_phrases", []),
                "example_counters":         c.get("example_counter_phrases", []),
                "message_count_in_cluster": c.get("message_count"),
                "top_keywords":             c.get("top_keywords", [])[:6],
                "date_range":               c.get("date_range"),
            }
            for i, c in enumerate(sensitive_clusters)
        ],
        "all_cluster_results": all_results,
    }
    return report


def print_summary(report: Dict):
    """Print a human-readable summary to stdout."""
    print("\n" + "="*70)
    print("  DLP INTENT SUGGESTION REPORT")
    print("="*70)
    s = report["summary"]
    print(f"  Tenant:              {report['tenant_id']}")
    print(f"  Total messages:      {s['total_messages']}")
    print(f"  Already detected:    {s['messages_with_existing_detection']}")
    print(f"  Undetected (input):  {s['messages_without_detection']}")
    print(f"  Clusters analyzed:   {s['clusters_analyzed']}")
    print(f"  Sensitive clusters:  {s['sensitive_clusters_found']}")
    print()

    if report["existing_detections"]:
        print("─── Currently Firing Detections ──────────────────────────────────")
        for name, count in report["existing_detections"].items():
            print(f"  [{count:>4}x]  {name}")
        print()

    if not report["suggested_new_models"]:
        print("  ✅  No new DLP models suggested — no undetected sensitive clusters found.")
    else:
        print("─── Suggested New DLP Models ──────────────────────────────────────")
        for m in report["suggested_new_models"]:
            print(f"\n  #{m['rank']}  [{m['risk_level'].upper()}]  {m['suggested_dlp_model']}")
            print(f"       Category:   {m['intent_category']}")
            print(f"       Confidence: {m['confidence']:.0%}")
            print(f"       Why:        {m['intent_description']}")
            print(f"       Keywords:   {', '.join(m['top_keywords'])}")
            print(f"       Messages:   {m['message_count_in_cluster']} in cluster")
            if m["example_triggers"]:
                print(f"       Examples:   {m['example_triggers'][0][:80]}…")
            if m["example_counters"]:
                print(f"       Counters:   {m['example_counters'][0][:80]}…")
    print("\n" + "="*70 + "\n")



# Main

def main():
    parser = argparse.ArgumentParser(description="DLP Intent Suggestion POC (Enhanced)")
    parser.add_argument("--input", required=True, help="Path to JSON log file")
    parser.add_argument("--output", default=f"dlp_suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", help="Output JSON path")
    parser.add_argument("--max-clusters", type=int, default=None, help="Max clusters to send to OpenAI (default: all). Larger clusters are prioritized.")
    # Clustering parameters
    parser.add_argument("--eps", type=float, default=DEFAULT_DBSCAN_EPS, help="DBSCAN eps (cosine distance)")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_DBSCAN_MIN_SAMPLES, help="DBSCAN min_samples / HDBSCAN min_cluster_size")
    parser.add_argument("--clustering-algorithm", choices=['dbscan', 'hdbscan'], default='dbscan', help="Clustering algorithm to use")
    # UMAP options
    parser.add_argument("--use-umap", action='store_true', help="Apply UMAP dimensionality reduction before clustering")
    parser.add_argument("--umap-components", type=int, default=50, help="Number of UMAP components")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    # Deduplication options
    parser.add_argument("--dedup-method", choices=['jaccard', 'minhash'], default='jaccard', help="Near-duplicate detection method")
    parser.add_argument("--minhash-threshold", type=float, default=0.8, help="MinHash LSH similarity threshold (if using minhash)")
    parser.add_argument("--jaccard-threshold", type=float, default=0.92, help="Jaccard similarity threshold for shingle-based dedup")
    # Other options
    parser.add_argument("--min-token-length", type=int, default=DEFAULT_MIN_TOKEN_LENGTH, help="Minimum word count to keep a message")
    parser.add_argument("--representatives", type=int, default=DEFAULT_REPRESENTATIVES_N, help="Number of representative messages per cluster")
    parser.add_argument("--keywords", type=int, default=DEFAULT_KEYWORDS_N, help="Number of TF-IDF keywords per cluster")

    args = parser.parse_args()

    # Override global defaults with command line
    global DEDUP_THRESHOLD, MIN_TOKEN_LENGTH, REPRESENTATIVES_N, KEYWORDS_N
    DEDUP_THRESHOLD = args.jaccard_threshold
    MIN_TOKEN_LENGTH = args.min_token_length
    REPRESENTATIVES_N = args.representatives
    KEYWORDS_N = args.keywords

    
    # Load and split
    
    detected, undetected = load_and_split(args.input)
    tenant_id = (detected + undetected)[0].get("tenant_id", "unknown") if (detected or undetected) else "unknown"
    total = len(detected) + len(undetected)

    if not undetected:
        log.info("No undetected messages found. Nothing to cluster.")
        return

    
    # Preprocess (with deduplication)
    
    cleaned = preprocess(undetected, dedup_method=args.dedup_method, minhash_threshold=args.minhash_threshold)

    if len(cleaned) < args.min_samples:
        log.warning(f"Only {len(cleaned)} messages after preprocessing — not enough to cluster (min_samples={args.min_samples}).")
        return

    
    # Embed
    
    embeddings = embed(cleaned)

    
    # Optional UMAP reduction
    
    if args.use_umap:
        embeddings = reduce_dimensions(embeddings,
                                       n_components=args.umap_components,
                                       n_neighbors=args.umap_neighbors,
                                       min_dist=args.umap_min_dist)

    
    # Cluster
    
    labels = cluster(embeddings,
                     algorithm=args.clustering_algorithm,
                     eps=args.eps,
                     min_samples=args.min_samples)

    
    # Build cluster payloads
    
    payloads = build_cluster_payloads(cleaned, embeddings, labels)

    
    # LLM Analysis (if credentials available)
    
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        log.warning("Azure credentials not set. Skipping LLM analysis. Set env vars:")
        log.warning("  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT")
        # Dump raw cluster payloads so you can inspect them
        with open(args.output, "w") as f:
            json.dump(payloads, f, indent=2, default=_json_default)
        log.info(f"Raw cluster payloads saved to {args.output}")
        return

    # Limit clusters sent to OpenAI (prioritize larger clusters)
    if args.max_clusters is not None and args.max_clusters < len(payloads):
        payloads_sorted = sorted(payloads, key=lambda p: p.get("message_count", 0), reverse=True)
        payloads = payloads_sorted[: args.max_clusters]
        log.info(f"Limited to {len(payloads)} clusters (--max-clusters={args.max_clusters})")

    all_results = analyze_all_clusters(payloads)

    
    # Build & save report
    
    report = build_report(tenant_id, all_results, detected, len(cleaned), total)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)

    log.info(f"Report saved to {args.output}")
    print_summary(report)


if __name__ == "__main__":
    main()