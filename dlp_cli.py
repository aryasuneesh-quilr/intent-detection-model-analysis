"""
    # Intent analysis (undetected message clustering)
    python dlp_cli.py intent --input logs.json

    # FP analysis (detected message false-positive evaluation)
    python dlp_cli.py fp --input logs.json

    # Both analyses in one run
    python dlp_cli.py both --input logs.json

    # Flags
    --debug: Enable DEBUG logging (shows raw LLM responses)
    --max-clusters: Maximum number of clusters to send to OpenAI (default: all). Larger clusters are prioritized.
    --eps: DBSCAN eps (cosine distance)
    --min-samples: DBSCAN min_samples / HDBSCAN min_cluster_size
    --clustering-algorithm: Clustering algorithm to use (default: dbscan) [options: dbscan, hdbscan]
    --use-umap: Apply UMAP dimensionality reduction before clustering
    --umap-components: Number of UMAP components
    --umap-neighbors: UMAP n_neighbors
    --umap-min-dist: UMAP min_dist
    --dedup-method: Near-duplicate detection method (jaccard or minhash) (default: jaccard) [options: jaccard, minhash]
    --jaccard-threshold: Jaccard similarity threshold for shingle-based dedup
    --minhash-threshold: MinHash LSH similarity threshold (if using minhash)
    --min-token-length: Minimum word count to keep a message
    --representatives: Number of representatives to send to OpenAI
    --keywords: Number of keywords to send to OpenAI
    --samples-per-category: Number of samples to send to OpenAI per category
    --min-occurrences: Minimum number of occurrences to send to OpenAI per category
    --max-categories: Maximum number of categories to send to OpenAI
    --output: Output JSON path (auto-named if omitted)
    --input: Path to JSON log file

─── File mode (existing behaviour, unchanged) ────────────────────────────────
    python dlp_cli.py intent --input logs.json
    python dlp_cli.py fp     --input logs.json
    python dlp_cli.py both   --input logs.json

─── API mode (fetch directly from the logs dashboard) ───────────────────────
    # Credentials from .env / environment variables (recommended)
    python dlp_cli.py intent --api
    python dlp_cli.py fp     --api
    python dlp_cli.py both   --api

    # Override look-back window (default: 72 h)
    python dlp_cli.py both --api --lookback-hours 24
    python dlp_cli.py both --api --lookback-hours 168   # 1 week

    # Explicit Unix-epoch timestamps (float seconds)
    python dlp_cli.py both --api --start-ts 1772777880 --end-ts 1773037080

    # Override any credential inline (useful for ad-hoc tests)
    python dlp_cli.py both --api --quilr-tenant-id <uuid> --quilr-auth <token>

─── Environment variables for API mode ──────────────────────────────────────
    QUILR_BASE_URL      (default: https://dlp-platform.quilr.ai)
    QUILR_AUTH          shared auth token
    QUILR_USERNAME      API username
    QUILR_PASSWORD      API password
    QUILR_TENANT_ID     tenant UUID
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Dict

from dlp_analysis import DLPAnalysisEngine, DLPConfig, _json_default

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Console printers
# ──────────────────────────────────────────────────────────────────────────────

_RISK_COLORS = {"high_fp": "🔴", "moderate_fp": "🟡", "low_fp": "🟢", "no_fp": "✅"}


def print_intent_report(report: Dict):
    if "error" in report:
        print(f"\n  ⚠️  {report['error']}\n")
        return
    print("\n" + "=" * 70)
    print("  DLP INTENT SUGGESTION REPORT")
    print("=" * 70)
    s = report["summary"]
    print(f"  Tenant:              {report['tenant_id']}")
    print(f"  Total messages:      {s['total_messages']}")
    print(f"  Already detected:    {s['messages_with_existing_detection']}")
    print(f"  Undetected (raw):    {s['messages_without_detection']}")
    # Show preprocessing funnel if present (added in funnel-fix session)
    if "dropped_in_preprocessing" in s:
        print(f"  Dropped (preproc):   {s['dropped_in_preprocessing']}"
              f"  (short / near-duplicate messages removed)")
        print(f"  Clustered:           {s['clustered_messages']}")
    print(f"  Clusters analyzed:   {s['clusters_analyzed']}")
    print(f"  Sensitive clusters:  {s['sensitive_clusters_found']}")
    print()

    if report.get("existing_detections"):
        print("─── Currently Firing Detections ──────────────────────────────────")
        for name, count in report["existing_detections"].items():
            print(f"  [{count:>4}x]  {name}")
        print()

    models = report.get("suggested_new_models", [])
    if not models:
        print("  ✅  No new DLP models suggested.")
    else:
        print("─── Suggested New DLP Models ──────────────────────────────────────")
        for m in models:
            print(f"\n  #{m['rank']}  [{m['risk_level'].upper()}]  {m['suggested_dlp_model']}")
            print(f"       Category:   {m['intent_category']}")
            print(f"       Confidence: {m['confidence']:.0%}")
            if m.get("confidence_rationale"):
                print(f"       Conf why:   {m['confidence_rationale']}")
            print(f"       Why:        {m['intent_description']}")
            if m.get("risk_rationale"):
                print(f"       Risk why:   {m['risk_rationale']}")
            print(f"       Keywords:   {', '.join(m['top_keywords'])}")
            print(f"       Messages:   {m['message_count_in_cluster']} in cluster")
            if m.get("evidence_messages"):
                print(f"       Evidence:   {m['evidence_messages'][0][:100]}…")
            elif m.get("example_triggers"):
                print(f"       Example:    {m['example_triggers'][0][:80]}…")
            if m.get("example_counters"):
                print(f"       Counter:    {m['example_counters'][0][:80]}…")
    print("\n" + "=" * 70 + "\n")


def print_fp_report(report: Dict):
    if "error" in report:
        print(f"\n  ⚠️  {report['error']}\n")
        return
    print("\n" + "=" * 74)
    print("  DLP FALSE POSITIVE ANALYSIS REPORT")
    print("=" * 74)
    s = report["summary"]
    print(f"  Tenant:               {report['tenant_id']}")
    print(f"  Total messages:       {s['total_messages']}")
    print(f"  With detections:      {s['messages_with_detection']}")
    print(f"  Categories analyzed:  {s['categories_analyzed']}")
    print(f"  High FP categories:   {s['high_fp_categories']}")
    print(f"  Moderate FP:          {s['moderate_fp_categories']}")
    print()

    action = report.get("action_required", [])
    if not action:
        print("  ✅  All categories within acceptable FP rates.")
    else:
        print("─── Categories Requiring Action ─────────────────────────────────────")
        for item in action:
            icon   = _RISK_COLORS.get(item["assessment"], "⚪")
            fp_pct = f"{item['estimated_fp_rate']:.0%}" if item.get("estimated_fp_rate") is not None else "N/A"
            conf   = f"{item['fp_rate_confidence']:.0%}" if item.get("fp_rate_confidence") is not None else "N/A"
            print(f"\n  #{item['rank']}  {icon}  {item['category']}")
            print(f"       Type:             {item['detection_type']}")
            print(f"       FP Rate:          ~{fp_pct}  (confidence: {conf})")
            print(f"       Period totals:    {item['total_detections']} → "
                  f"~{item.get('estimated_tp_total','?')} TP / ~{item.get('estimated_fp_total','?')} FP")
            print(f"       Sample verdicts:  {item.get('sample_tp_count','?')} TP / "
                  f"{item.get('sample_fp_count','?')} FP")
            print(f"       Action:           {item['tuning_recommendation'].replace('_', ' ').upper()}")
            print(f"       FP pattern:       {item['fp_pattern_summary']}")
            if item.get("fp_rate_rationale"):
                print(f"       Rate rationale:   {item['fp_rate_rationale']}")
            print(f"       Rationale:        {item['tuning_rationale']}")
            if item.get("assessment_rationale"):
                print(f"       Assessment why:   {item['assessment_rationale']}")
            if item.get("suggested_allowlist"):
                print(f"       Allowlist:        {item['suggested_allowlist'][:3]}")
            print(f"       Risk if disabled: {item['risk_of_disabling'].upper()}")
            if item.get("key_fp_evidence"):
                ev = item["key_fp_evidence"][0]
                print(f"       Key evidence:     [{ev.get('verdict')}] "
                      f"{ev.get('input_snippet','')[:80]}…")

    print("\n─── All Categories ───────────────────────────────────────────────────")
    for r in report.get("all_category_results", []):
        icon    = _RISK_COLORS.get(r.get("overall_assessment"), "⚪")
        fp_pct  = f"{r.get('estimated_fp_rate', 0):.0%}"
        tp_t    = r.get("estimated_tp_count_total", "?")
        total_n = r.get("total_detections_in_period", "?")
        rec     = r.get("tuning_recommendation", "?")
        dtype   = r.get("dominant_detection_type", "?")
        print(f"  {icon}  {r['detection_category']:<45} [{dtype:<10}]  FP~{fp_pct:<5}  TP={tp_t}/{total_n}  [{rec}]")
    print("\n" + "=" * 74 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DLP Analysis CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mode", choices=["intent", "fp", "both"],
                        help="Analysis mode: intent clustering, FP analysis, or both")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (auto-named if omitted)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG logging")

    # ── Input source (mutually exclusive) ─────────────────────────────────────
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", metavar="FILE",
                     help="Path to a local JSON log file (file mode)")
    src.add_argument("--api", action="store_true",
                     help="Fetch records from the Quilr platform API (API mode)")

    # ── API mode: time window ──────────────────────────────────────────────────
    ag = parser.add_argument_group(
        "API mode — time window (--api only)",
        "Only one of --lookback-hours or --start-ts/--end-ts may be used. "
        "Default: 72 h look-back."
    )
    ag.add_argument("--lookback-hours", type=int, default=None,
                    metavar="N",
                    help="Fetch logs from the last N hours (default: 72)")
    ag.add_argument("--start-ts", type=float, default=None,
                    metavar="EPOCH",
                    help="Explicit window start (Unix epoch, float seconds)")
    ag.add_argument("--end-ts", type=float, default=None,
                    metavar="EPOCH",
                    help="Explicit window end (Unix epoch, float seconds; default: now)")

    # ── API mode: credential overrides ────────────────────────────────────────
    cg = parser.add_argument_group(
        "API mode — credential overrides (--api only)",
        "These override the corresponding environment variables for this run only. "
        "Omit to rely on .env / shell exports."
    )
    cg.add_argument("--quilr-base-url",    default=None, metavar="URL")
    cg.add_argument("--quilr-auth",        default=None, metavar="TOKEN")
    cg.add_argument("--quilr-username",    default=None, metavar="USER")
    cg.add_argument("--quilr-password",    default=None, metavar="PASS")
    cg.add_argument("--quilr-tenant-id",   default=None, metavar="UUID")
    cg.add_argument("--skip-tenant-check", action="store_true",
                    help="Skip the /tenant_config preflight call")

    # ── Intent analysis options ───────────────────────────────────────────────
    ig = parser.add_argument_group("Intent analysis options")
    ig.add_argument("--max-clusters",          type=int,   default=None)
    ig.add_argument("--eps",                   type=float, default=0.35)
    ig.add_argument("--min-samples",           type=int,   default=3)
    ig.add_argument("--clustering-algorithm",  choices=["dbscan", "hdbscan"], default="dbscan")
    ig.add_argument("--use-umap",              action="store_true")
    ig.add_argument("--umap-components",       type=int,   default=50)
    ig.add_argument("--umap-neighbors",        type=int,   default=15)
    ig.add_argument("--umap-min-dist",         type=float, default=0.1)
    ig.add_argument("--dedup-method",          choices=["jaccard", "minhash"], default="jaccard")
    ig.add_argument("--jaccard-threshold",     type=float, default=0.92)
    ig.add_argument("--minhash-threshold",     type=float, default=0.8)
    ig.add_argument("--min-token-length",      type=int,   default=15)
    ig.add_argument("--representatives",       type=int,   default=5)
    ig.add_argument("--keywords",              type=int,   default=10)

    # ── FP analysis options ───────────────────────────────────────────────────
    fg = parser.add_argument_group("FP analysis options")
    fg.add_argument("--max-categories",        type=int, default=None)
    fg.add_argument("--samples-per-category",  type=int, default=8)
    fg.add_argument("--min-occurrences",       type=int, default=2)

    return parser


def _validate_api_args(args):
    """Extra cross-argument validation for API mode."""
    if not args.api:
        return
    if args.lookback_hours is not None and \
       (args.start_ts is not None or args.end_ts is not None):
        print("error: --lookback-hours cannot be combined with --start-ts / --end-ts",
              file=sys.stderr)
        sys.exit(2)


def _save(report: Dict, path: str):
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    log.info(f"Report saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()
    _validate_api_args(args)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Build config (common + any Quilr credential overrides) ───────────────
    config_kwargs = dict(
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
        clustering_algorithm=args.clustering_algorithm,
        use_umap=args.use_umap,
        umap_components=args.umap_components,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        dedup_method=args.dedup_method,
        dedup_threshold=args.jaccard_threshold,
        minhash_threshold=args.minhash_threshold,
        min_token_length=args.min_token_length,
        representatives_n=args.representatives,
        keywords_n=args.keywords,
        fp_samples_per_category=args.samples_per_category,
        fp_min_occurrences=args.min_occurrences,
    )
    # Only inject credential overrides when they were explicitly supplied,
    # so that .env values remain in effect when args are None.
    if args.api:
        if args.quilr_base_url:  config_kwargs["quilr_base_url"]  = args.quilr_base_url
        if args.quilr_auth:      config_kwargs["quilr_auth"]      = args.quilr_auth
        if args.quilr_username:  config_kwargs["quilr_username"]  = args.quilr_username
        if args.quilr_password:  config_kwargs["quilr_password"]  = args.quilr_password
        if args.quilr_tenant_id: config_kwargs["quilr_tenant_id"] = args.quilr_tenant_id

    config = DLPConfig(**config_kwargs)
    engine = DLPAnalysisEngine(config)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Fetch records (file or API) ───────────────────────────────────────────
    if args.api:
        print(f"\n  📡  API mode — fetching from Quilr platform…")
        if args.lookback_hours:
            print(f"       Look-back:  {args.lookback_hours} hours")
        elif args.start_ts:
            from datetime import datetime as _dt
            s = _dt.utcfromtimestamp(args.start_ts).strftime("%Y-%m-%d %H:%M UTC")
            e = _dt.utcfromtimestamp(args.end_ts).strftime("%Y-%m-%d %H:%M UTC") \
                if args.end_ts else "now"
            print(f"       Window:     {s} → {e}")
        else:
            print(f"       Look-back:  72 hours (default)")

        # Fetch once — reused for both intent and FP if mode == "both"
        records = engine.quilr.validate_and_fetch(
            start_ts=args.start_ts,
            end_ts=args.end_ts,
            lookback_hours=args.lookback_hours,
            skip_tenant_check=args.skip_tenant_check,
        )
        print(f"       Fetched:    {len(records):,} records\n")
    else:
        records = engine.loader.load(args.input)

    # ── Run analysis ──────────────────────────────────────────────────────────
    if args.mode in ("intent", "both"):
        report = engine.run_intent_analysis(records, max_clusters=args.max_clusters)
        path   = args.output or f"dlp_intent_{ts}.json"
        _save(report, path)
        print_intent_report(report)

    if args.mode in ("fp", "both"):
        report = engine.run_fp_analysis(records, max_categories=args.max_categories)
        path   = args.output if args.mode == "fp" and args.output \
                 else f"dlp_fp_{ts}.json"
        _save(report, path)
        print_fp_report(report)


if __name__ == "__main__":
    main()