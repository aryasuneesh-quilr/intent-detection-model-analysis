"""
Install:
    pip install sentence-transformers scikit-learn numpy requests python-dotenv
Optional:
    pip install umap-learn hdbscan datasketch
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import hdbscan as hdbscan_lib
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False

load_dotenv()

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _word_shingles(text: str, k: int = 3) -> Set[str]:
    words = text.lower().split()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _jaccard(a: Set, b: Set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ──────────────────────────────────────────────────────────────────────────────
# Audit Logger
# ──────────────────────────────────────────────────────────────────────────────

class DLPLogger:
    """
    Writes a structured JSONL audit trail to <log_dir>/dlp_audit_<timestamp>.jsonl.

    Every entry has:
        ts          — ISO timestamp
        event       — event type string
        **payload   — event-specific fields

    Event types
    -----------
    llm_call        — every LLM request/response (prompt, raw output, parsed result, latency_ms, tokens)
    cluster_built   — one entry per cluster after build_payloads() (id, size, keywords, representatives)
    cluster_summary — one entry at end of clustering summarising label distribution
    fp_category     — one entry per FP category payload sent to the LLM
    run_start / run_end — bracket each top-level analysis run
    """

    def __init__(self, log_dir: str = "dlp_logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts        = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"dlp_audit_{ts}.jsonl")
        self._fh  = open(self.path, "a", encoding="utf-8")
        log.info(f"Audit log → {self.path}")

    def _write(self, event: str, **payload):
        entry = {"ts": datetime.utcnow().isoformat() + "Z", "event": event, **payload}
        self._fh.write(json.dumps(entry, default=_json_default) + "\n")
        self._fh.flush()

    def log_llm_call(self, *, context: str, system_prompt: str, user_prompt: str,
                     raw_response: str, parsed: Optional[Dict],
                     latency_ms: float, finish_reason: str, attempt: int):
        self._write(
            "llm_call",
            context=context,
            attempt=attempt,
            finish_reason=finish_reason,
            latency_ms=round(latency_ms, 1),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_response=raw_response,
            parsed_ok=parsed is not None,
            parsed=parsed,
        )

    def log_cluster(self, *, cluster_id, message_count: int,
                    keywords: List[str], representatives: List[str]):
        self._write(
            "cluster_built",
            cluster_id=cluster_id,
            message_count=message_count,
            keywords=keywords,
            representatives=representatives,
        )

    def log_cluster_summary(self, *, algorithm: str, n_clusters: int,
                            n_noise: int, label_counts: Dict):
        self._write(
            "cluster_summary",
            algorithm=algorithm,
            n_clusters=n_clusters,
            n_noise=n_noise,
            label_counts=label_counts,
        )

    def log_fp_category(self, *, category: str, dominant_type: str,
                        total_detections: int, samples_shown: int,
                        mean_fp_prior: float, payload: Dict):
        self._write(
            "fp_category",
            category=category,
            dominant_type=dominant_type,
            total_detections=total_detections,
            samples_shown=samples_shown,
            mean_fp_prior=mean_fp_prior,
            payload=payload,
        )

    def log_run(self, event: str, **kwargs):
        self._write(event, **kwargs)

    def close(self):
        self._fh.close()


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DLPConfig:
    # Azure
    endpoint:    str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    api_key:     str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"))
    deployment:  str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"))

    # Rate limiting
    rate_limit_calls:  int   = 8
    rate_limit_period: float = 60.0

    # Intent clustering
    embedding_model:   str   = "all-MiniLM-L6-v2"
    dbscan_eps:        float = 0.35
    dbscan_min_samples: int  = 3
    min_token_length:  int   = 15
    dedup_threshold:   float = 0.92
    representatives_n: int   = 5
    keywords_n:        int   = 10
    clustering_algorithm: str = "dbscan"  # "dbscan" | "hdbscan"
    use_umap:          bool  = False
    umap_components:   int   = 50
    umap_neighbors:    int   = 15
    umap_min_dist:     float = 0.1
    dedup_method:      str   = "jaccard"  # "jaccard" | "minhash"
    minhash_threshold: float = 0.8

    # FP analysis
    fp_samples_per_category:  int   = 8
    fp_min_occurrences:       int   = 2
    entity_context_window:    int   = 300
    max_snippet_length:       int   = 1200
    pattern_fp_prior:         float = 0.6
    contextual_fp_prior:      float = 0.2

    # Logging
    log_dir: str = "dlp_logs"

    def headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json", "api-key": self.api_key}

    def chat_url(self) -> str:
        return (
            f"{self.endpoint}/openai/deployments/{self.deployment}"
            f"/chat/completions?api-version={self.api_version}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Azure OpenAI Client
# ──────────────────────────────────────────────────────────────────────────────

class AzureOpenAIClient:
    def __init__(self, config: DLPConfig, dlp_logger: Optional["DLPLogger"] = None):
        self.config     = config
        self.dlp_logger = dlp_logger
        self._timestamps: List[float] = []

    def _rate_limit(self):
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < self.config.rate_limit_period]
        if len(self._timestamps) >= self.config.rate_limit_calls:
            sleep_for = self.config.rate_limit_period - (now - self._timestamps[0])
            if sleep_for > 0:
                log.info(f"Rate limit: sleeping {sleep_for:.1f}s")
                time.sleep(sleep_for)
        self._timestamps.append(time.time())

    @staticmethod
    def _extract_json(raw: str) -> Optional[str]:
        raw = raw.strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and start < end:
            candidate = raw[start:end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            return None

    # Token multipliers applied on retry: attempt 1 = 1×, attempt 2 = 1.5×, attempt 3 = 2×
    _TOKEN_MULTIPLIERS = {1: 1.0, 2: 1.5, 3: 2.0}

    def chat(self, system_prompt: str, user_prompt: str,
             temperature: float = 0.1, max_tokens: int = 3000,
             context: str = "unknown") -> str:
        self._rate_limit()
        last_failure_reason = "unknown"

        for attempt in range(1, 4):
            # Escalate token budget on retries so truncated responses get room to breathe
            tokens_this_attempt = int(max_tokens * self._TOKEN_MULTIPLIERS[attempt])
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens":  tokens_this_attempt,
            }
            if attempt > 1:
                log.info(f"Retry attempt {attempt}/3 for [{context}] "
                         f"(max_tokens={tokens_this_attempt}, reason={last_failure_reason!r})")

            t0 = time.time()
            try:
                r = requests.post(
                    self.config.chat_url(),
                    headers=self.config.headers(),
                    json=payload,
                    timeout=90,
                )
                r.raise_for_status()
                choice        = r.json()["choices"][0]
                finish_reason = choice.get("finish_reason", "")
                raw_content   = choice.get("message", {}).get("content")
                latency_ms    = (time.time() - t0) * 1000

                log.debug(f"LLM response (attempt {attempt}, finish={finish_reason!r}):\n{raw_content}")

                if not raw_content:
                    last_failure_reason = "empty_response"
                    raise ValueError("Empty LLM response")

                extracted = self._extract_json(raw_content)
                parsed    = None
                if extracted is not None:
                    try:
                        parsed = json.loads(extracted)
                    except json.JSONDecodeError:
                        pass

                if self.dlp_logger:
                    self.dlp_logger.log_llm_call(
                        context=context,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        raw_response=raw_content or "",
                        parsed=parsed,
                        latency_ms=latency_ms,
                        finish_reason=finish_reason,
                        attempt=attempt,
                    )

                if extracted is None:
                    # Distinguish truncation (needs more tokens) from a genuine bad response
                    last_failure_reason = "truncated_no_json" if finish_reason == "length" else "no_json_in_response"
                    log.warning(f"[{context}] No JSON in response "
                                f"(attempt {attempt}, finish={finish_reason!r}, "
                                f"tokens={tokens_this_attempt}). "
                                f"Preview:\n{raw_content[:400]}")
                    raise ValueError(last_failure_reason)

                return extracted

            except (requests.HTTPError, requests.Timeout) as e:
                latency_ms = (time.time() - t0) * 1000
                last_failure_reason = type(e).__name__
                log.error(f"[{context}] LLM call error (attempt {attempt}/3): {e}")
                if self.dlp_logger:
                    self.dlp_logger.log_llm_call(
                        context=context,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        raw_response=str(e),
                        parsed=None,
                        latency_ms=latency_ms,
                        finish_reason="error",
                        attempt=attempt,
                    )

            except ValueError as e:
                latency_ms = (time.time() - t0) * 1000
                # Already warned above; just log and loop
                if self.dlp_logger:
                    self.dlp_logger.log_llm_call(
                        context=context,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        raw_response=str(e),
                        parsed=None,
                        latency_ms=latency_ms,
                        finish_reason=last_failure_reason,
                        attempt=attempt,
                    )

            if attempt < 3:
                time.sleep(2 ** attempt)

        # All three attempts exhausted — always raise RuntimeError so callers
        # can catch it uniformly and skip this item without crashing the run.
        raise RuntimeError(
            f"LLM exhausted 3 attempts for [{context}]. "
            f"Last failure: {last_failure_reason}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Log Loader
# ──────────────────────────────────────────────────────────────────────────────

class LogLoader:
    @staticmethod
    def load(path: str) -> List[Dict]:
        content, last_err = None, None
        for enc, err in [("utf-8", "strict"), ("utf-8-sig", "strict"),
                         ("utf-8", "replace"), ("latin-1", "strict")]:
            try:
                with open(path, "r", encoding=enc, errors=err) as f:
                    content = f.read()
                break
            except UnicodeDecodeError as e:
                last_err = e
        if content is None:
            raise last_err

        try:
            data = json.loads(content)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array, got {type(data).__name__}")
            return data
        except json.JSONDecodeError as e:
            log.warning(f"JSON parse error — attempting recovery: {e}")
            return LogLoader._recover(content, e)

    @staticmethod
    def _recover(content: str, original: json.JSONDecodeError) -> List[Dict]:
        a, b = content.find("["), content.rfind("]")
        if a == -1 or b == -1:
            raise original
        body = content[a + 1:b]
        records, depth, in_str = [], 0, False
        escape, quote, obj_start, skipped = False, None, None, 0
        for i, ch in enumerate(body):
            if escape:
                escape = False; continue
            if ch == "\\" and in_str:
                escape = True; continue
            if ch in ('"', "'") and not in_str:
                in_str, quote = True, ch; continue
            if ch == quote and in_str:
                in_str, quote = False, None; continue
            if in_str:
                continue
            if ch == "{":
                if depth == 0: obj_start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and obj_start is not None:
                    try:
                        obj = json.loads(body[obj_start:i + 1])
                        if isinstance(obj, dict): records.append(obj)
                    except json.JSONDecodeError:
                        skipped += 1
                    obj_start = None
        log.info(f"Recovered {len(records)} records (skipped {skipped})")
        return records

    @staticmethod
    def split(records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        detected, undetected = [], []
        for r in records:
            if not r.get("input_text") or r.get("errors"):
                continue
            (detected if r.get("response_jsonl") else undetected).append(r)
        log.info(f"{len(records)} records → {len(detected)} detected, {len(undetected)} undetected")
        return detected, undetected


# ──────────────────────────────────────────────────────────────────────────────
# Intent Analyzer
# ──────────────────────────────────────────────────────────────────────────────

_INTENT_SYSTEM_PROMPT = (
    "You are a Data Loss Prevention (DLP) expert. Analyze clusters of user prompts "
    "sent to an AI chatbot and identify sensitive data patterns the customer's DLP "
    "system has not configured detection for. "
    "Return ONLY valid JSON matching the schema in the user message."
)

_INTENT_ANALYSIS_PROMPT = """\
You are analyzing a cluster of user prompts sent to an AI chatbot by employees of a company.
No DLP detection model was triggered for any of these prompts.

Determine whether this cluster represents a sensitive data pattern the company should detect.
If yes, propose a new DLP detection model for it.

Cluster data:
{cluster_json}

NAMING GUIDELINES (apply if sensitive):
- intent_name: short, specific, noun-phrase label for what is being shared/requested.
  Good: "Employee Salary Band Disclosure", "AWS Secret Key Exposure", "Patient Diagnosis Sharing"
  Bad:  "Sensitive Data", "PII", "Financial Info"
- intent_description: 1-2 sentences starting with "This intent covers...". Describe the
  specific data type and the user behaviour (e.g. sharing, requesting, pasting, exporting).
- suggested_dlp_model: the detection model name a DLP engineer would recognise.
  Mirror the intent_name but as a detector label, e.g. "Employee Compensation Data Detector".

*IMPORTANT* - EXAMPLE GUIDELINES:
- example_triggering_phrases: generate AS MANY as needed to cover the full range of surface
  forms seen in the cluster. Include direct shares, indirect references, paraphrases, and
  domain jargon variants. Aim for breadth — a sparse positive set causes missed detections.
- example_counter_phrases: generate AT LEAST as many counter-examples as triggering phrases.
  These must be plausible, superficially similar phrases that are NOT actually sensitive.
  Good counters share vocabulary or structure with triggers but lack the sensitive payload.
  Example — trigger: "my AWS_SECRET_KEY is AKIAIOSFODNN7EXAMPLE"
             counter: "the config key for dark mode is theme_secret_key=dark"
  The counter set is used to train the boundary of the model; sparse counters cause FPs.

*IMPORTANT* - EVIDENCE AND SCORING GUIDELINES:
These guidelines ensure that confidence scores and risk levels are traceable to the actual
input data you have been given, not to generic assumptions.

- evidence_messages: Select 2–4 verbatim excerpts (≤200 characters each) from the
  representative_messages in the cluster data above that most directly demonstrate why
  this cluster is (or is not) sensitive. These should be the specific passages of text
  that, if seen by a DLP analyst, would immediately trigger an alert. Do not paraphrase
  or synthesise — copy directly from the representative_messages provided.
  Return [] if is_sensitive=false.

- confidence_rationale: Write 1–2 sentences explaining exactly which features of the
  evidence_messages drove the sensitivity_confidence score you assigned.
  Be specific: name the data elements (e.g. "salary figures", "internal IP addresses",
  "CVE identifiers with CVSS scores") that were present and how consistently they appeared
  across the cluster. Do NOT give a generic explanation.
  Example: "9 of the representative messages contained explicit CVE IDs paired with CVSS
  scores ≥9.0 and patch-status language, leaving no ambiguity about the sensitivity type."

- risk_rationale: Write 1–2 sentences grounding the risk_level assignment in what the
  evidence_messages actually contain. Name the worst-case exfiltration scenario directly
  supported by the data in this cluster — do not rely on general category-level reasoning.
  Example: "The messages include full employee IDs, joining dates, and itemised salary
  components; exfiltration would allow precise financial profiling of named individuals."

- sensitivity_confidence (0.0–1.0): Must reflect the density and clarity of sensitive
  signals in evidence_messages. Use the following anchors:
    0.95–1.0 — Unambiguous sensitive data present in the majority of messages
               (e.g. verbatim PII, credentials, CVEs with scores)
    0.80–0.94 — Clear sensitive pattern but some messages in the cluster are ambiguous
    0.60–0.79 — Probable sensitivity but content is indirect or partially obfuscated
    < 0.60   — Weak signal; the cluster may be borderline or context-dependent
  Do not assign 0.95+ unless evidence_messages strongly support it.

- risk_level: Must be grounded in what could realistically happen if this data were
  exfiltrated. Use the following anchors:
    critical — Data that directly enables system compromise, fraud, or mass harm
               (e.g. credentials, CVEs with active exploits, PHI at scale)
    high     — Data that enables targeted harm to individuals or serious operational damage
               (e.g. salary data, internal asset inventories, audit findings)
    medium   — Data that is embarrassing or operationally sensitive but limited in blast radius
    low      — Data that is mildly sensitive or context-dependent

Respond with ONLY valid JSON, no markdown, no extra text:
{{
  "cluster_id": "<id from input>",
  "is_sensitive": true | false,
  "sensitivity_confidence": <0.0–1.0>,
  "confidence_rationale": "<1-2 sentences grounding the score in specific evidence, or null>",
  "intent_name": "<specific noun-phrase label, or null>",
  "detected_intent_category": "<free-form category string inferred from content, or null>",
  "intent_description": "<1-2 sentences starting with 'This intent covers...', or null>",
  "suggested_dlp_model": "<detector label or null>",
  "suggested_model_rationale": "<why this model is needed, or null>",
  "evidence_messages": ["<verbatim excerpt ≤200 chars from representative_messages>", "..."],
  "example_triggering_phrases": ["<phrase>", "..."],
  "example_counter_phrases": ["<phrase>", "..."],
  "risk_level": "low" | "medium" | "high" | "critical",
  "risk_rationale": "<1-2 sentences grounding the risk_level in what the data actually contains, or null>"
}}

If not sensitive: is_sensitive=false, set evidence_messages=[], and set all other fields null
except cluster_id, is_sensitive, sensitivity_confidence, and confidence_rationale.
Risk level = potential impact if this data were exfiltrated.
"""


class IntentAnalyzer:
    def __init__(self, config: DLPConfig, client: AzureOpenAIClient,
                 dlp_logger: Optional["DLPLogger"] = None):
        self.cfg        = config
        self.client     = client
        self.dlp_logger = dlp_logger
        self._model: Optional[SentenceTransformer] = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            log.info(f"Loading embedding model: {self.cfg.embedding_model}")
            self._model = SentenceTransformer(self.cfg.embedding_model)
        return self._model

    # ── Preprocessing ────────────────────────────────────────────────────────

    def preprocess(self, records: List[Dict]) -> List[Dict]:
        normalized, seen_shingles = [], []
        for r in records:
            text = _normalize_text(r["input_text"])
            if len(text.split()) < self.cfg.min_token_length:
                continue
            r = dict(r)
            r["clean_text"] = text
            normalized.append(r)

        if not normalized:
            return []

        texts = [r["clean_text"] for r in normalized]
        if self.cfg.dedup_method == "minhash" and DATASKETCH_AVAILABLE:
            indices = self._minhash_dedup(texts)
        else:
            indices = self._jaccard_dedup(texts)

        cleaned = [normalized[i] for i in indices]
        log.info(f"Preprocessed: {len(cleaned)} unique messages from {len(records)}")
        return cleaned

    def _jaccard_dedup(self, texts: List[str]) -> List[int]:
        seen, indices = [], []
        for i, text in enumerate(texts):
            shingles = _word_shingles(text)
            if not any(_jaccard(shingles, s) >= self.cfg.dedup_threshold for s in seen):
                seen.append(shingles)
                indices.append(i)
        log.info(f"Jaccard dedup: {len(indices)}/{len(texts)} kept")
        return indices

    def _minhash_dedup(self, texts: List[str]) -> List[int]:
        lsh       = MinHashLSH(threshold=self.cfg.minhash_threshold, num_perm=128)
        minhashes = []
        for text in texts:
            m = MinHash(num_perm=128)
            for w in text.lower().split():
                m.update(w.encode("utf8"))
            minhashes.append(m)
        indices = []
        for i, m in enumerate(minhashes):
            if not lsh.query(m):
                lsh.insert(f"doc_{i}", m)
                indices.append(i)
        log.info(f"MinHash dedup: {len(indices)}/{len(texts)} kept")
        return indices

    # ── Embedding + Clustering ────────────────────────────────────────────────

    def embed(self, records: List[Dict]) -> np.ndarray:
        texts = [r["clean_text"] for r in records]
        log.info(f"Embedding {len(texts)} messages…")
        return self._get_model().encode(texts, batch_size=64,
                                        show_progress_bar=True, convert_to_numpy=True)

    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.cfg.use_umap or not UMAP_AVAILABLE:
            return embeddings
        log.info(f"UMAP: {embeddings.shape[1]}→{self.cfg.umap_components} dims")
        reducer = umap.UMAP(
            n_components=self.cfg.umap_components,
            n_neighbors=self.cfg.umap_neighbors,
            min_dist=self.cfg.umap_min_dist,
            random_state=42,
        )
        return reducer.fit_transform(embeddings)

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        X = normalize(embeddings)
        if self.cfg.clustering_algorithm == "hdbscan" and HDBSCAN_AVAILABLE:
            algo = "hdbscan"
            log.info(f"HDBSCAN (min_cluster_size={self.cfg.dbscan_min_samples})")
            labels = hdbscan_lib.HDBSCAN(
                min_cluster_size=self.cfg.dbscan_min_samples,
                metric="euclidean",
            ).fit_predict(X)
        else:
            algo = "dbscan"
            log.info(f"DBSCAN (eps={self.cfg.dbscan_eps}, min_samples={self.cfg.dbscan_min_samples})")
            labels = DBSCAN(
                eps=self.cfg.dbscan_eps,
                min_samples=self.cfg.dbscan_min_samples,
                metric="cosine",
                n_jobs=-1,
            ).fit(X).labels_

        unique, counts  = np.unique(labels, return_counts=True)
        label_counts    = {int(k): int(v) for k, v in zip(unique, counts)}
        n_clusters      = sum(1 for k in label_counts if k != -1)
        n_noise         = label_counts.get(-1, 0)
        log.info(f"Clusters: {n_clusters}, noise: {n_noise}")

        if self.dlp_logger:
            self.dlp_logger.log_cluster_summary(
                algorithm=algo,
                n_clusters=n_clusters,
                n_noise=n_noise,
                label_counts=label_counts,
            )
        return labels

    # ── Payload Building ─────────────────────────────────────────────────────

    def build_payloads(self, records: List[Dict],
                       embeddings: np.ndarray, labels: np.ndarray) -> List[Dict]:
        groups: Dict[int, List[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            groups[label].append(i)

        payloads = []
        for label, indices in sorted(groups.items()):
            recs  = [records[i] for i in indices]
            embs  = embeddings[indices]
            texts = [r["clean_text"] for r in recs]
            ts    = sorted(r.get("created_at", "") for r in recs if r.get("created_at"))
            kws   = self._tfidf_keywords(texts)
            reps  = self._representatives(recs, embs)
            cid   = label if label != -1 else "noise"

            if self.dlp_logger:
                self.dlp_logger.log_cluster(
                    cluster_id=cid,
                    message_count=len(recs),
                    keywords=kws,
                    representatives=reps,
                )

            payloads.append({
                "cluster_id":              cid,
                "message_count":           len(recs),
                "date_range":              {"from": ts[0] if ts else None, "to": ts[-1] if ts else None},
                "top_keywords":            kws,
                "representative_messages": reps,
            })
        return payloads

    def _tfidf_keywords(self, texts: List[str]) -> List[str]:
        if len(texts) < 2:
            freq: Dict[str, int] = defaultdict(int)
            for w in " ".join(texts).lower().split():
                if len(w) > 3: freq[w] += 1
            return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:self.cfg.keywords_n]]
        try:
            tfidf  = TfidfVectorizer(stop_words="english", max_features=300, ngram_range=(1, 2))
            X      = tfidf.fit_transform(texts)
            scores = np.asarray(X.mean(axis=0)).flatten()
            terms  = tfidf.get_feature_names_out()
            return [t for t, _ in sorted(zip(terms, scores), key=lambda x: -x[1])[:self.cfg.keywords_n]]
        except Exception:
            return []

    def _representatives(self, recs: List[Dict], embs: np.ndarray) -> List[str]:
        centroid  = embs.mean(axis=0, keepdims=True)
        distances = cosine_distances(centroid, embs).flatten()
        return [recs[i]["clean_text"] for i in distances.argsort()[:self.cfg.representatives_n]]

    # ── LLM Analysis ─────────────────────────────────────────────────────────

    def analyze_cluster(self, payload: Dict) -> Optional[Dict]:
        prompt = _INTENT_ANALYSIS_PROMPT.format(
            cluster_json=json.dumps(payload, indent=2, default=_json_default)
        )
        try:
            raw    = self.client.chat(_INTENT_SYSTEM_PROMPT, prompt, temperature=0.1,
                                      max_tokens=2200, context=f"intent:cluster:{payload['cluster_id']}")
            result = json.loads(raw)
            result.update({
                "message_count": payload["message_count"],
                "top_keywords":  payload["top_keywords"],
                "date_range":    payload["date_range"],
            })
            return result
        except (json.JSONDecodeError, RuntimeError) as e:
            log.warning(f"Cluster {payload['cluster_id']} analysis failed: {e}")
            return None

    def analyze_all(self, payloads: List[Dict],
                    max_clusters: Optional[int] = None) -> List[Dict]:
        # Sort by message_count descending so the most-populated clusters go first.
        ordered = sorted(payloads, key=lambda p: -p["message_count"])

        if max_clusters and len(ordered) > max_clusters:
            # Primary queue (will be worked through first) + overflow reserve
            primary  = list(ordered[:max_clusters])
            overflow = list(ordered[max_clusters:])
        else:
            primary  = list(ordered)
            overflow = []

        results = []
        worked  = 0
        target  = len(primary)  # how many successful results we want

        while primary:
            payload = primary.pop(0)
            worked += 1
            log.info(f"Cluster {payload['cluster_id']} "
                     f"({worked}/{target + len(overflow)}, queue={len(primary)} left)…")
            result = self.analyze_cluster(payload)

            if result:
                results.append(result)
            else:
                # This cluster failed — pull the next best from overflow to keep
                # the quota filled, if a limit was requested.
                if max_clusters and overflow:
                    replacement = overflow.pop(0)
                    primary.append(replacement)
                    log.info(f"Cluster {payload['cluster_id']} skipped — "
                             f"queuing replacement cluster {replacement['cluster_id']} "
                             f"(overflow remaining: {len(overflow)})")
                else:
                    log.warning(f"Cluster {payload['cluster_id']} skipped — no overflow available.")

        sensitive = sum(1 for r in results if r.get("is_sensitive"))
        log.info(f"Intent analysis: {sensitive}/{len(results)} sensitive "
                 f"(skipped {worked - len(results)} failures)")
        return results

    # ── Report ────────────────────────────────────────────────────────────────

    def build_report(self, tenant_id: str, all_results: List[Dict],
                     detected: List[Dict], undetected_raw: int,
                     cleaned_count: int, total: int) -> Dict:
        """
        Parameters
        ----------
        undetected_raw : len(undetected) — messages with no existing detection,
                         before any preprocessing (the true "input" count).
        cleaned_count  : len(cleaned)    — messages that survived preprocess()
                         (min_token_length filter + dedup); this is what was
                         actually clustered.
        """
        existing: Dict[str, int] = defaultdict(int)
        for r in detected:
            for det in r.get("response_jsonl", []):
                existing[det.get("name", "Unknown")] += 1

        sensitive = sorted(
            [r for r in all_results if r.get("is_sensitive")],
            key=lambda x: -x.get("sensitivity_confidence", 0),
        )
        return {
            "tenant_id":    tenant_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "total_messages":                   total,
                "messages_with_existing_detection": len(detected),
                # ── undetected funnel ──────────────────────────────────────
                # messages_without_detection : raw split — no existing detection
                # dropped_in_preprocessing   : removed by min_token_length / dedup
                # clustered_messages         : what DBSCAN/HDBSCAN actually saw
                "messages_without_detection":       undetected_raw,
                "dropped_in_preprocessing":         undetected_raw - cleaned_count,
                "clustered_messages":               cleaned_count,
                # ──────────────────────────────────────────────────────────
                "clusters_analyzed":                len(all_results),
                "sensitive_clusters_found":         len(sensitive),
            },
            "existing_detections": dict(existing),
            "suggested_new_models": [
                {
                    "rank":                     i + 1,
                    "cluster_id":               c["cluster_id"],
                    "intent_name":              c.get("intent_name"),
                    "risk_level":               c.get("risk_level"),
                    "risk_rationale":           c.get("risk_rationale"),
                    "confidence":               c.get("sensitivity_confidence"),
                    "confidence_rationale":     c.get("confidence_rationale"),
                    "intent_category":          c.get("detected_intent_category"),
                    "intent_description":       c.get("intent_description"),
                    "suggested_dlp_model":      c.get("suggested_dlp_model"),
                    "rationale":                c.get("suggested_model_rationale"),
                    "evidence_messages":        c.get("evidence_messages", []),
                    "example_triggers":         c.get("example_triggering_phrases", []),
                    "example_counters":         c.get("example_counter_phrases", []),
                    "message_count_in_cluster": c.get("message_count"),
                    "top_keywords":             c.get("top_keywords", [])[:6],
                    "date_range":               c.get("date_range"),
                }
                for i, c in enumerate(sensitive)
            ],
            "all_cluster_results": all_results,
        }


# ──────────────────────────────────────────────────────────────────────────────
# FP Analyzer
# ──────────────────────────────────────────────────────────────────────────────

_FP_SYSTEM_PROMPT = (
    "You are a DLP expert and ML engineer evaluating whether automated DLP detections "
    "are genuine (true positives) or incorrect (false positives). "
    "Return ONLY valid JSON matching the schema in the user message."
)

_FP_ANALYSIS_PROMPT = """\
Evaluate false positives for DLP detection category: "{category}"
dominant_detection_type: "{det_type}"

Each example includes:
- input_text: user prompt (windowed around entities; "---" = window boundary)
- full_detection: complete detection result
  * sensitive_entities_positive_examples: strings the system flagged (NOT verified ground truth)
  * entity_texts_with_subcategories: sub-type per entity (e.g., "UPI ID", "NATIONAL ID NUMBER")
  * detection_type: "pattern" (regex) or "contextual" (ML model)
  * reason: system explanation

HOW TO EVALUATE:
  1. Read input_text carefully — understand what the user is actually doing or asking.
  2. For each sensitive_entities_positive_example, ask:
       - Is this string actually a "{category}" in this context?
       - Does the sub-type label make sense? (e.g., is an @email.com address really a UPI ID?)
       - Would a trained DLP analyst flag this as a genuine policy violation?
  3. Consider domain context: healthcare text may contain IDs that look like numbers;
     finance text may contain codes; IT text may contain hashes or tokens.
  4. A detection is a TRUE POSITIVE if the entity is genuinely sensitive in this context.
     A detection is a FALSE POSITIVE if the entity matched a pattern but is not actually sensitive.

- - -
TUNING RECOMMENDATION RULES (strictly enforced):
  dominant_detection_type for this category = "{det_type}"

  If dominant_detection_type is "pattern":
    → ONLY use: "disable", "tighten_regex", or "no_action"
    → NEVER use "add_allowlist" or "reduce_scope"
    → tighten_regex: the regex pattern needs to be narrowed or made more specific

  If dominant_detection_type is "contextual":
    → ONLY use: "disable", "add_allowlist", "reduce_scope", or "no_action"
    → NEVER use "tighten_regex"
    → add_allowlist: exclude known-safe patterns/domains from triggering this model
    → reduce_scope: narrow the detection to specific data types or contexts

  Shared options:
    → disable:   FPs are so overwhelming that the category provides little genuine signal
    → no_action: FP rate is within acceptable range; category is working correctly

  suggested_allowlist_patterns: regex or literal strings to add as allowlist exclusions (negative examples for the detection model).
  Add AS MANY as needed to cover the full range of surface forms seen in the category. Aim for breadth — a sparse positive set causes missed detections.
  Example: ["@veg", "\\.com$", "\\b[A-Z]{{1}}\\d{{7}}\\b"] - these are examples of patterns that are not sensitive in this context.
  Return [] if not applicable.
- - -

*IMPORTANT* - EVIDENCE AND SCORING GUIDELINES:
These guidelines ensure that fp_rate, confidence, and overall_assessment are traceable to
the actual input examples you have been given, not to generic assumptions.

- per_example_verdicts: For each example, you MUST populate:
    * input_snippet: first 120 characters of input_text (verbatim)
    * flagged_entity: the exact string from sensitive_entities_positive_examples that
      drove your verdict (copy verbatim from the example; use the most significant one
      if multiple entities are present)
    * entity_subcategory: the sub-type label from entity_texts_with_subcategories for
      that entity (e.g. "UPI ID", "AADHAAR", "NATIONAL ID NUMBER")
    * verdict: "TP", "FP", or "UNCERTAIN"
    * reason: name the specific entity and explain in one sentence why it is or is not
      genuinely sensitive in context — reference the surrounding input_text, not just
      the entity in isolation

- key_fp_evidence: After completing per_example_verdicts, select the 2–3 examples whose
  flagged entities most strongly illustrate the dominant FP pattern for this category.
  For each, provide:
    * input_snippet: first 120 characters of input_text (verbatim, same as above)
    * flagged_entity: verbatim entity string that was flagged
    * entity_subcategory: the sub-type label
    * verdict: your verdict for this example
    * fp_reason: one sentence explaining precisely why this entity is a false positive
      (or true positive) — reference what the surrounding text reveals about context

- estimated_fp_rate (0.0–1.0): Compute this directly from per_example_verdicts:
    fp_rate = (count of FP verdicts) / (total verdicts excluding UNCERTAIN).
  If all verdicts are UNCERTAIN, use 0.5.
  Do NOT rely on generic category-level priors — derive it from what you actually saw.

- fp_rate_confidence (0.0–1.0): Reflects how clearly the sample supports your fp_rate estimate.
  Use the following anchors:
    0.85–1.0  — All or nearly all verdicts are unambiguous (clear TP or clear FP)
    0.60–0.84 — Most verdicts are clear but some are UNCERTAIN or borderline
    0.40–0.59 — Many verdicts are UNCERTAIN; the sample is noisy or mixed
    < 0.40    — Majority of verdicts are UNCERTAIN; estimate is speculative

- fp_rate_rationale: Write 1–2 sentences explaining how you arrived at estimated_fp_rate.
  Name the specific pattern(s) in the flagged entities that made them FPs or TPs.
  Example: "6 of 8 flagged entities were @domain email addresses miscategorised as UPI IDs
  because the pattern matched the '@' character; the 2 TPs were genuine UPI VPA strings
  of the form name@upibank."

- overall_assessment: Derive from estimated_fp_rate using these thresholds:
    high_fp     → estimated_fp_rate > 0.60
    moderate_fp → estimated_fp_rate 0.30–0.60
    low_fp      → estimated_fp_rate 0.10–0.30
    no_fp       → estimated_fp_rate < 0.10
  Do not assign high_fp unless the evidence in key_fp_evidence clearly supports it.

- assessment_rationale: Write 1–2 sentences grounding the overall_assessment in the
  specific evidence you found. Reference what the flagged entities actually were and why
  the majority were or were not genuine.

Detection category data:
{payload_json}

Respond with ONLY valid JSON, no markdown, no extra text:
{{
  "detection_category": "<category name>",
  "dominant_detection_type": "{det_type}",
  "estimated_fp_rate": <float 0.0–1.0>,
  "fp_rate_confidence": <float 0.0–1.0>,
  "fp_rate_rationale": "<1-2 sentences deriving fp_rate from per_example_verdicts>",
  "estimated_tp_count_in_sample": <int>,
  "estimated_fp_count_in_sample": <int>,
  "fp_pattern_summary": "<1-2 sentences: what is causing FPs, or null>",
  "tuning_recommendation": "disable" | "tighten_regex" | "add_allowlist" | "reduce_scope" | "no_action",
  "tuning_rationale": "<1-2 sentences>",
  "per_example_verdicts": [
    {{
      "input_snippet": "<first 120 chars of input_text, verbatim>",
      "flagged_entity": "<verbatim entity string that drove the verdict>",
      "entity_subcategory": "<sub-type label from entity_texts_with_subcategories>",
      "verdict": "TP" | "FP" | "UNCERTAIN",
      "reason": "<one sentence: name the entity and explain why it is/isn't sensitive in context>"
    }}
  ],
  "key_fp_evidence": [
    {{
      "input_snippet": "<first 120 chars of input_text, verbatim>",
      "flagged_entity": "<verbatim entity string>",
      "entity_subcategory": "<sub-type label>",
      "verdict": "TP" | "FP" | "UNCERTAIN",
      "fp_reason": "<one sentence: why this entity is or is not a genuine detection>"
    }}
  ],
  "suggested_allowlist_patterns": ["<pattern>"],
  "risk_of_disabling": "low" | "medium" | "high",
  "overall_assessment": "high_fp" | "moderate_fp" | "low_fp" | "no_fp",
  "assessment_rationale": "<1-2 sentences grounding overall_assessment in key_fp_evidence>"
}}

Thresholds: high_fp >60%, moderate_fp 30-60%, low_fp 10-30%, no_fp <10%
"""


class FPAnalyzer:
    def __init__(self, config: DLPConfig, client: AzureOpenAIClient,
                 dlp_logger: Optional["DLPLogger"] = None):
        self.cfg        = config
        self.client     = client
        self.dlp_logger = dlp_logger

    # ── Grouping ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_det_type(detection: Dict) -> str:
        raw = (detection.get("detection_type") or "").lower().strip()
        if raw == "pattern":
            return "pattern"
        if raw in ("contextual", "semantic", "model", "ml"):
            return "contextual"
        if raw == "edm":
            return "edm"
        return "unknown"

    def group_by_category(self, detected: List[Dict]) -> Dict[str, List[Dict]]:
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for record in detected:
            for det in record.get("response_jsonl", []):
                cat      = det.get("name") or det.get("code_name") or "Unknown"
                det_type = self._normalize_det_type(det)
                fp_prior = (self.cfg.pattern_fp_prior if det_type == "pattern"
                            else self.cfg.contextual_fp_prior)
                groups[cat].append({
                    "record":         record,
                    "detection":      det,
                    "detection_type": det_type,
                    "fp_prior":       fp_prior,
                })
        for cat, items in groups.items():
            p = sum(1 for i in items if i["detection_type"] == "pattern")
            c = sum(1 for i in items if i["detection_type"] == "contextual")
            log.info(f"'{cat}': {len(items)} detections (pattern={p}, contextual={c})")
        return dict(groups)

    @staticmethod
    def _dominant_type(items: List[Dict]) -> str:
        counts: Dict[str, int] = defaultdict(int)
        for item in items:
            counts[item["detection_type"]] += 1
        return max(counts, key=lambda k: counts[k])

    # ── Scoring + Sampling ───────────────────────────────────────────────────

    def _fp_score(self, item: Dict) -> float:
        score = item["fp_prior"]
        det   = item["detection"]
        rec   = item["record"]

        for entity, subcat in det.get("entity_texts_with_subcategories", {}).items():
            if re.match(r"[^@]+@[^@]+", entity) and "upi" in subcat.lower():
                score = min(score + 0.35, 1.0)
            if re.match(r"^[\w.-]+@[\w-]+$", entity) and not re.search(r"\.(com|org|net|io|ai)$", entity):
                score = min(score + 0.2, 1.0)

        if len((rec.get("input_text") or "").split()) < 20:
            score = min(score + 0.15, 1.0)

        entities = det.get("sensitive_entities", [])
        if len(entities) > 3 and len(set(entities)) <= 2:
            score = min(score + 0.15, 1.0)

        return round(score, 3)

    def sample(self, items: List[Dict]) -> Tuple[List[Dict], float]:
        for item in items:
            item["heuristic_fp_score"] = self._fp_score(item)
        sorted_items = sorted(items, key=lambda x: -x["heuristic_fp_score"])
        mean_prior   = float(np.mean([i["heuristic_fp_score"] for i in items]))
        return sorted_items[:self.cfg.fp_samples_per_category], mean_prior

    # ── Snippet Building ─────────────────────────────────────────────────────

    def _entity_snippet(self, text: str, entities: List[str]) -> str:
        if not text:
            return ""
        if not entities:
            return self._head_tail(text)

        text_lower = text.lower()
        windows: List[List[int]] = []
        for entity in entities:
            if not entity:
                continue
            el, pos = entity.lower().strip(), 0
            while True:
                idx = text_lower.find(el, pos)
                if idx == -1:
                    break
                ws = max(0, idx - self.cfg.entity_context_window)
                we = min(len(text), idx + len(entity) + self.cfg.entity_context_window)
                windows.append([ws, we])
                pos = idx + 1

        if not windows:
            return self._head_tail(text)

        windows.sort()
        merged: List[List[int]] = []
        for ws, we in windows:
            if merged and ws <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], we)
            else:
                merged.append([ws, we])

        snippets, used = [], 0
        for ws, we in merged:
            chunk = ("…" if ws > 0 else "") + text[ws:we] + ("…" if we < len(text) else "")
            rem   = self.cfg.max_snippet_length - used
            if len(chunk) > rem:
                if rem > 60:
                    snippets.append(chunk[:rem] + "…")
                break
            snippets.append(chunk)
            used += len(chunk)
        return "\n---\n".join(snippets)

    def _head_tail(self, text: str) -> str:
        if len(text) <= self.cfg.max_snippet_length:
            return text
        half = self.cfg.max_snippet_length // 2 - 15
        return text[:half] + "\n…[middle truncated]…\n" + text[-half:]

    # ── Payload Building ─────────────────────────────────────────────────────

    def build_payload(self, category: str, sampled: List[Dict],
                      total: int, mean_prior: float, dominant_type: str) -> Dict:
        examples = []
        for item in sampled:
            rec = item["record"]
            det = item["detection"]
            entities = det.get("sensitive_entities", [])
            examples.append({
                "input_text":          self._entity_snippet(rec.get("input_text") or "", entities),
                "input_was_truncated": len(rec.get("input_text") or "") > self.cfg.max_snippet_length,
                "full_detection": {
                    "id":             det.get("id"),
                    "name":           det.get("name"),
                    "type":           det.get("type"),
                    "detection_type": det.get("detection_type"),
                    "reason":         det.get("reason"),
                    "sensitive_entities_positive_examples": entities,
                    "entity_texts_with_subcategories":      det.get("entity_texts_with_subcategories", {}),
                },
                "heuristic_fp_score": item["heuristic_fp_score"],
            })
        return {
            "detection_category":      category,
            "dominant_detection_type": dominant_type,
            "total_detections":        total,
            "samples_shown":           len(examples),
            "mean_heuristic_fp_prior": round(mean_prior, 3),
            "examples":                examples,
        }

    # ── LLM Analysis ─────────────────────────────────────────────────────────

    def analyze_category(self, payload: Dict) -> Optional[Dict]:
        if self.dlp_logger:
            self.dlp_logger.log_fp_category(
                category=payload["detection_category"],
                dominant_type=payload["dominant_detection_type"],
                total_detections=payload["total_detections"],
                samples_shown=payload["samples_shown"],
                mean_fp_prior=payload["mean_heuristic_fp_prior"],
                payload=payload,
            )
        prompt = _FP_ANALYSIS_PROMPT.format(
            category=payload["detection_category"],
            det_type=payload["dominant_detection_type"],
            payload_json=json.dumps(payload, indent=2, default=_json_default),
        )
        try:
            raw    = self.client.chat(_FP_SYSTEM_PROMPT, prompt, temperature=0.1,
                                      max_tokens=3500, context=f"fp:category:{payload['detection_category']}")
            result = json.loads(raw)
            total_n = payload["total_detections"]
            fp_rate = result.get("estimated_fp_rate") or 0.0
            result["total_detections_in_period"] = total_n
            result["estimated_fp_count_total"]   = round(fp_rate * total_n)
            result["estimated_tp_count_total"]   = total_n - round(fp_rate * total_n)
            return result
        except (json.JSONDecodeError, RuntimeError) as e:
            log.warning(f"FP analysis failed for '{payload['detection_category']}': {e}")
            return None

    def analyze_all(self, groups: Dict[str, List[Dict]],
                    max_categories: Optional[int] = None) -> List[Dict]:
        # Build full ranked list (by mean FP prior, highest first)
        all_payloads = []
        for cat, items in groups.items():
            if len(items) < self.cfg.fp_min_occurrences:
                log.info(f"Skipping '{cat}' ({len(items)} occurrences < min={self.cfg.fp_min_occurrences})")
                continue
            sampled, mean_prior = self.sample(items)
            dominant            = self._dominant_type(items)
            all_payloads.append((cat, self.build_payload(cat, sampled, len(items), mean_prior, dominant), mean_prior))

        all_payloads.sort(key=lambda x: -x[2])

        if max_categories and len(all_payloads) > max_categories:
            primary  = list(all_payloads[:max_categories])
            overflow = list(all_payloads[max_categories:])
        else:
            primary  = list(all_payloads)
            overflow = []

        log.info(f"Analyzing {len(primary)} FP categories "
                 f"({len(overflow)} in overflow reserve)…")

        results = []
        worked  = 0

        while primary:
            cat, payload, prior = primary.pop(0)
            worked += 1
            log.info(f"'{cat}' ({worked}, queue={len(primary)} left) "
                     f"[{payload['dominant_detection_type']}] prior={prior:.2f}")
            result = self.analyze_category(payload)

            if result:
                result["mean_heuristic_fp_prior"] = round(prior, 3)
                results.append(result)
            else:
                if max_categories and overflow:
                    replacement = overflow.pop(0)
                    primary.append(replacement)
                    log.info(f"Category '{cat}' skipped — "
                             f"queuing replacement '{replacement[0]}' "
                             f"(overflow remaining: {len(overflow)})")
                else:
                    log.warning(f"Category '{cat}' skipped — no overflow available.")

        results.sort(key=lambda x: -x.get("estimated_fp_rate", 0))
        log.info(f"FP analysis complete: {len(results)} categories "
                 f"(skipped {worked - len(results)} failures)")
        return results

    # ── Report ────────────────────────────────────────────────────────────────

    _ASSESSMENT_ORDER = {"high_fp": 0, "moderate_fp": 1, "low_fp": 2, "no_fp": 3}

    def build_report(self, tenant_id: str, results: List[Dict],
                     detected_count: int, undetected_count: int, total: int) -> Dict:
        high     = [r for r in results if r.get("overall_assessment") == "high_fp"]
        moderate = [r for r in results if r.get("overall_assessment") == "moderate_fp"]
        sorted_results = sorted(
            results, key=lambda x: self._ASSESSMENT_ORDER.get(x.get("overall_assessment", "no_fp"), 4)
        )
        return {
            "tenant_id":    tenant_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "total_messages":             total,
                "messages_with_detection":    detected_count,
                "messages_without_detection": undetected_count,
                "categories_analyzed":        len(results),
                "high_fp_categories":         len(high),
                "moderate_fp_categories":     len(moderate),
            },
            "action_required": [
                {
                    "rank":                    i + 1,
                    "category":                r["detection_category"],
                    "detection_type":          r.get("dominant_detection_type"),
                    "assessment":              r.get("overall_assessment"),
                    "assessment_rationale":    r.get("assessment_rationale"),
                    "estimated_fp_rate":       r.get("estimated_fp_rate"),
                    "fp_rate_confidence":      r.get("fp_rate_confidence"),
                    "fp_rate_rationale":       r.get("fp_rate_rationale"),
                    "total_detections":        r.get("total_detections_in_period"),
                    "estimated_tp_total":      r.get("estimated_tp_count_total"),
                    "estimated_fp_total":      r.get("estimated_fp_count_total"),
                    "sample_tp_count":         r.get("estimated_tp_count_in_sample"),
                    "sample_fp_count":         r.get("estimated_fp_count_in_sample"),
                    "fp_pattern_summary":      r.get("fp_pattern_summary"),
                    "key_fp_evidence":         r.get("key_fp_evidence", []),
                    "tuning_recommendation":   r.get("tuning_recommendation"),
                    "tuning_rationale":        r.get("tuning_rationale"),
                    "suggested_allowlist":     r.get("suggested_allowlist_patterns", []),
                    "risk_of_disabling":       r.get("risk_of_disabling"),
                    "mean_heuristic_fp_prior": r.get("mean_heuristic_fp_prior"),
                }
                for i, r in enumerate(sorted_results)
                if r.get("overall_assessment") in ("high_fp", "moderate_fp")
            ],
            "all_category_results": results,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Top-level Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class DLPAnalysisEngine:
    def __init__(self, config: Optional[DLPConfig] = None):
        self.config     = config or DLPConfig()
        self.dlp_logger = DLPLogger(self.config.log_dir)
        self.client     = AzureOpenAIClient(self.config, self.dlp_logger)
        self.intent     = IntentAnalyzer(self.config, self.client, self.dlp_logger)
        self.fp         = FPAnalyzer(self.config, self.client, self.dlp_logger)
        self.loader     = LogLoader()

    def run_intent_analysis(
        self,
        records: List[Dict],
        max_clusters: Optional[int] = None,
    ) -> Dict:
        detected, undetected = LogLoader.split(records)
        tenant_id = (detected + undetected)[0].get("tenant_id", "unknown") if (detected or undetected) else "unknown"
        total     = len(detected) + len(undetected)

        self.dlp_logger.log_run("run_start", mode="intent", tenant_id=tenant_id,
                                total_records=total, max_clusters=max_clusters)

        if not undetected:
            self.dlp_logger.log_run("run_end", mode="intent", status="no_undetected")
            return {"error": "No undetected messages found"}

        cleaned    = self.intent.preprocess(undetected)
        if len(cleaned) < self.config.dbscan_min_samples:
            self.dlp_logger.log_run("run_end", mode="intent", status="too_few_messages",
                                    cleaned_count=len(cleaned))
            return {"error": f"Only {len(cleaned)} messages after preprocessing — not enough to cluster"}

        embeddings = self.intent.embed(cleaned)
        embeddings = self.intent.reduce(embeddings)
        labels     = self.intent.cluster(embeddings)
        payloads   = self.intent.build_payloads(cleaned, embeddings, labels)
        results    = self.intent.analyze_all(payloads, max_clusters=max_clusters)
        report     = self.intent.build_report(
                         tenant_id, results, detected,
                         undetected_raw=len(undetected),
                         cleaned_count=len(cleaned),
                         total=total,
                     )

        self.dlp_logger.log_run("run_end", mode="intent", status="ok",
                                clusters_analyzed=len(results),
                                sensitive_found=report["summary"]["sensitive_clusters_found"])
        return report

    def run_fp_analysis(
        self,
        records: List[Dict],
        max_categories: Optional[int] = None,
    ) -> Dict:
        detected, undetected = LogLoader.split(records)
        all_records = detected + undetected
        tenant_id   = all_records[0].get("tenant_id", "unknown") if all_records else "unknown"

        self.dlp_logger.log_run("run_start", mode="fp", tenant_id=tenant_id,
                                detected_count=len(detected), max_categories=max_categories)

        if not detected:
            self.dlp_logger.log_run("run_end", mode="fp", status="no_detected")
            return {"error": "No detected messages found"}

        groups  = self.fp.group_by_category(detected)
        results = self.fp.analyze_all(groups, max_categories=max_categories)
        report  = self.fp.build_report(tenant_id, results, len(detected), len(undetected), len(all_records))

        self.dlp_logger.log_run("run_end", mode="fp", status="ok",
                                categories_analyzed=len(results),
                                high_fp=report["summary"]["high_fp_categories"])
        return report

    def load_and_run_intent(self, path: str, max_clusters: Optional[int] = None) -> Dict:
        return self.run_intent_analysis(LogLoader.load(path), max_clusters=max_clusters)

    def load_and_run_fp(self, path: str, max_categories: Optional[int] = None) -> Dict:
        return self.run_fp_analysis(LogLoader.load(path), max_categories=max_categories)