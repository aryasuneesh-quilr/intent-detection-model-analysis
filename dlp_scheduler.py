#!/usr/bin/env python3
"""
dlp_scheduler.py — Self-contained scheduler for automated DLP analysis
Key behaviours 
  • State file: last successful run time is written to --state-file so the
    scheduler is restart-safe.  If restarted within the 72 h window it will
    sleep out the remainder rather than double-fetching.
  • First run: runs immediately on first start (no wait), then every 72 h.
  • Graceful shutdown: SIGTERM / Ctrl-C finishes the current run first.
  • Log rotation: optional --log-file rotates nightly, keeps 30 days.

 Quick start 
    python dlp_scheduler.py                          # both analyses, 72-h cadence
    python dlp_scheduler.py --mode intent            # intent only
    python dlp_scheduler.py --mode fp                # FP only
    python dlp_scheduler.py --dry-run                # print config and exit
    python dlp_scheduler.py --run-once               # one shot and exit

 Changing the cadence 
    python dlp_scheduler.py --interval-hours 48      # every 2 days
    python dlp_scheduler.py --interval-hours 168     # every 7 days

    NOTE: --interval-hours and --lookback-hours are independent.  Keep them
    equal (both 72) to get non-overlapping windows and avoid double-counting.
    Increasing the interval while keeping lookback=72 creates gaps; decreasing
    it creates overlap (more API calls, redundant data).

 Running as a system service (recommended for production) 
    # systemd — create /etc/systemd/system/dlp-scheduler.service:
    #   [Unit]
    #   Description=DLP Analysis Scheduler
    #   After=network.target
    #
    #   [Service]
    #   WorkingDirectory=</path/to/project>
    #   ExecStart=/usr/bin/python3 dlp_scheduler.py --log-file /var/log/dlp/scheduler.log
    #   Restart=on-failure
    #   RestartSec=60
    #   EnvironmentFile=/path/to/project/.env
    #
    #   [Install]
    #   WantedBy=multi-user.target

 Using as a real cron entry instead 
    Use --run-once so the script exits after each invocation; cron handles
    the schedule:
        0 2 */3 * *  cd /path/to/project && python dlp_scheduler.py --run-once >> /var/log/dlp/cron.log 2>&1

 All flags 
    See --help for the full list.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

from dlp_analysis import DLPAnalysisEngine, DLPConfig, _json_default

log = logging.getLogger("dlp_scheduler")

# Default cadence — 3 days expressed in hours
_DEFAULT_INTERVAL_HOURS = 72
_DEFAULT_LOOKBACK_HOURS = 72

# Graceful shutdown

_SHUTDOWN = False

def _handle_signal(sig, frame):
    global _SHUTDOWN
    log.info(f"Signal {sig} received — finishing current run then stopping.")
    _SHUTDOWN = True

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)

# State file  (restart-safe "last run" tracking)

def _read_state(state_file: Path) -> Optional[float]:
    """Return the Unix timestamp of the last successful run, or None."""
    try:
        data = json.loads(state_file.read_text())
        return float(data["last_successful_run"])
    except Exception:
        return None


def _write_state(state_file: Path, ts: float):
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({
        "last_successful_run": ts,
        "last_successful_run_human": datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }, indent=2))


def _seconds_until_next_run(state_file: Path, interval_s: float) -> float:
    """
    How many seconds to sleep before the next run is due.
    Returns 0 if a run is overdue or no state file exists.
    """
    last = _read_state(state_file)
    if last is None:
        return 0.0
    elapsed = time.time() - last
    remaining = interval_s - elapsed
    return max(0.0, remaining)

# Argument parser  (all dlp_cli.py flags are present here)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DLP Analysis Scheduler — runs dlp_engine on a 3-day cadence "
            "via the Quilr platform API."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    #  Scheduler options 
    sg = parser.add_argument_group("Scheduler options")
    sg.add_argument(
        "--interval-hours", type=float, default=_DEFAULT_INTERVAL_HOURS,
        metavar="N",
        help=(
            "How often to run the analysis (hours). "
            f"Default {_DEFAULT_INTERVAL_HOURS}h = 3 days. "
            "Keep equal to --lookback-hours to avoid gaps or overlap."
        ),
    )
    sg.add_argument(
        "--mode", choices=["intent", "fp", "both"], default="both",
        help="Which analysis to run on each scheduled execution.",
    )
    sg.add_argument(
        "--run-once", action="store_true",
        help=(
            "Run the analysis exactly once then exit. "
            "State file is still written. Useful when cron/systemd handles "
            "the schedule externally."
        ),
    )
    sg.add_argument(
        "--dry-run", action="store_true",
        help="Print resolved configuration and exit without fetching or analysing.",
    )
    sg.add_argument(
        "--output-dir", default="dlp_reports", metavar="DIR",
        help="Directory where JSON reports are saved.",
    )
    sg.add_argument(
        "--state-file", default="dlp_scheduler_state.json", metavar="FILE",
        help=(
            "JSON file used to track the last successful run time. "
            "Lets the scheduler resume correctly after a restart without "
            "re-running immediately if the window hasn't elapsed."
        ),
    )
    sg.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG logging.",
    )
    sg.add_argument(
        "--log-file", default=None, metavar="FILE",
        help="Also write logs to this file (rotates nightly, keeps 30 days).",
    )
    sg.add_argument(
        "--no-state", action="store_true",
        help=(
            "Disable state file tracking. The scheduler will always run "
            "immediately on start regardless of when it last ran."
        ),
    )

    #  API / time-window options 
    apig = parser.add_argument_group(
        "API — time window",
        "The scheduler always uses API mode. The look-back window is "
        "independent of the interval — set both to 72 h for clean "
        "non-overlapping windows.",
    )
    apig.add_argument(
        "--lookback-hours", type=int, default=_DEFAULT_LOOKBACK_HOURS, metavar="N",
        help="How many hours of logs to fetch on each run.",
    )
    apig.add_argument(
        "--skip-tenant-check", action="store_true",
        help="Skip the /tenant_config preflight call on each run.",
    )

    #  API credential overrides 
    cg = parser.add_argument_group(
        "API — credential overrides",
        "All default to the corresponding environment variable / .env entry. "
        "Override here for a single deployment without editing .env.",
    )
    cg.add_argument("--quilr-base-url",  default=None, metavar="URL")
    cg.add_argument("--quilr-auth",      default=None, metavar="TOKEN")
    cg.add_argument("--quilr-username",  default=None, metavar="USER")
    cg.add_argument("--quilr-password",  default=None, metavar="PASS")
    cg.add_argument("--quilr-tenant-id", default=None, metavar="UUID")

    #  Intent options (mirrors dlp_cli.py exactly) 
    ig = parser.add_argument_group("Intent analysis options")
    ig.add_argument("--max-clusters",         type=int,   default=None)
    ig.add_argument("--eps",                  type=float, default=0.35)
    ig.add_argument("--min-samples",          type=int,   default=3)
    ig.add_argument("--clustering-algorithm", choices=["dbscan", "hdbscan"], default="dbscan")
    ig.add_argument("--use-umap",             action="store_true")
    ig.add_argument("--umap-components",      type=int,   default=50)
    ig.add_argument("--umap-neighbors",       type=int,   default=15)
    ig.add_argument("--umap-min-dist",        type=float, default=0.1)
    ig.add_argument("--dedup-method",         choices=["jaccard", "minhash"], default="jaccard")
    ig.add_argument("--jaccard-threshold",    type=float, default=0.92)
    ig.add_argument("--minhash-threshold",    type=float, default=0.8)
    ig.add_argument("--min-token-length",     type=int,   default=15)
    ig.add_argument("--representatives",      type=int,   default=5)
    ig.add_argument("--keywords",             type=int,   default=10)

    #  FP options (mirrors dlp_cli.py exactly) 
    fg = parser.add_argument_group("FP analysis options")
    fg.add_argument("--max-categories",       type=int, default=None)
    fg.add_argument("--samples-per-category", type=int, default=8)
    fg.add_argument("--min-occurrences",      type=int, default=2)

    return parser


# Logging setup

def setup_logging(debug: bool, log_file: Optional[str]):
    level = logging.DEBUG if debug else logging.INFO
    fmt   = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = TimedRotatingFileHandler(
            log_file, when="midnight", backupCount=30, encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter(fmt))
        handlers.append(fh)

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


# Config builder

def build_config(args) -> DLPConfig:
    kwargs = dict(
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
        quilr_default_lookback_hours=args.lookback_hours,
    )
    # Only inject credential overrides that were explicitly given on the CLI
    if args.quilr_base_url:  kwargs["quilr_base_url"]  = args.quilr_base_url
    if args.quilr_auth:      kwargs["quilr_auth"]      = args.quilr_auth
    if args.quilr_username:  kwargs["quilr_username"]  = args.quilr_username
    if args.quilr_password:  kwargs["quilr_password"]  = args.quilr_password
    if args.quilr_tenant_id: kwargs["quilr_tenant_id"] = args.quilr_tenant_id
    return DLPConfig(**kwargs)


# Save helper

def _save(report: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default)
    log.info(f"Report saved → {path}")


# Single analysis run

def run_analysis(args, engine: DLPAnalysisEngine, output_dir: Path) -> bool:
    """
    Fetches records and runs the configured analysis mode.
    Returns True on success, False on unrecoverable error.
    Records are fetched once and shared between intent and FP to avoid a
    second API call when mode == "both".
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info(
        f" Starting DLP analysis run [{ts}]  "
        f"mode={args.mode}  lookback={args.lookback_hours}h "
    )

    #  Fetch records (single call regardless of mode) 
    try:
        records = engine.quilr.validate_and_fetch(
            lookback_hours=args.lookback_hours,
            skip_tenant_check=args.skip_tenant_check,
        )
        log.info(f"Fetched {len(records):,} records from Quilr API.")
    except Exception as e:
        log.error(f"Failed to fetch records: {e}", exc_info=True)
        return False

    if not records:
        log.warning("API returned 0 records — skipping analysis for this run.")
        return True   # not an error; nothing to analyse

    #  Intent analysis 
    if args.mode in ("intent", "both"):
        try:
            report = engine.run_intent_analysis(
                records, max_clusters=args.max_clusters
            )
            path = output_dir / f"dlp_intent_{ts}.json"
            _save(report, path)
            s = report.get("summary", {})
            log.info(
                f"Intent done: total={s.get('total_messages')}  "
                f"undetected={s.get('messages_without_detection')}  "
                f"dropped={s.get('dropped_in_preprocessing')}  "
                f"clustered={s.get('clustered_messages')}  "
                f"sensitive={s.get('sensitive_clusters_found')}  "
                f"→ {path.name}"
            )
        except Exception as e:
            log.error(f"Intent analysis failed: {e}", exc_info=True)
            if args.mode == "intent":
                return False
            # In 'both' mode: log and continue to FP rather than aborting

    #  FP analysis 
    if args.mode in ("fp", "both"):
        try:
            report = engine.run_fp_analysis(
                records, max_categories=args.max_categories
            )
            path = output_dir / f"dlp_fp_{ts}.json"
            _save(report, path)
            s = report.get("summary", {})
            log.info(
                f"FP done: with_detections={s.get('messages_with_detection')}  "
                f"categories={s.get('categories_analyzed')}  "
                f"high_fp={s.get('high_fp_categories')}  "
                f"→ {path.name}"
            )
        except Exception as e:
            log.error(f"FP analysis failed: {e}", exc_info=True)
            return False

    log.info(f" Run complete [{ts}] ")
    return True


# Dry-run printer

def print_dry_run(args):
    state_file = Path(args.state_file)
    last_run   = _read_state(state_file)
    interval_s = args.interval_hours * 3600
    secs_left  = _seconds_until_next_run(state_file, interval_s) if not args.no_state else 0

    print("\n" + "=" * 66)
    print("  DLP SCHEDULER — DRY RUN (no fetch, no analysis)")
    print("=" * 66)
    print(f"  Mode:                {args.mode}")
    if args.run_once:
        print( "  Cadence:             run-once (--run-once)")
    else:
        print(f"  Cadence:             every {args.interval_hours}h")
    print(f"  Lookback window:     last {args.lookback_hours}h per run")
    print(f"  Output directory:    {args.output_dir}/")
    print(f"  State file:          {args.state_file}"
          + (" (disabled via --no-state)" if args.no_state else ""))
    if last_run and not args.no_state:
        last_dt = datetime.utcfromtimestamp(last_run).strftime("%Y-%m-%d %H:%M UTC")
        if secs_left > 0:
            h, rem  = divmod(int(secs_left), 3600)
            m       = rem // 60
            print(f"  Last run:            {last_dt}")
            print(f"  Next run due in:     {h}h {m}m")
        else:
            print(f"  Last run:            {last_dt}  ← overdue, would run immediately")
    elif not args.no_state:
        print( "  State file:          not found — would run immediately on start")
    print(f"  Skip tenant check:   {args.skip_tenant_check}")
    print()
    print("  Quilr credentials (from .env unless overridden):")
    for env_var, override in [
        ("QUILR_BASE_URL",  args.quilr_base_url),
        ("QUILR_AUTH",      args.quilr_auth),
        ("QUILR_USERNAME",  args.quilr_username),
        ("QUILR_PASSWORD",  args.quilr_password),
        ("QUILR_TENANT_ID", args.quilr_tenant_id),
    ]:
        source = "(CLI override)" if override else "(from env)"
        val    = override or os.getenv(env_var) or "⚠️  NOT SET"
        masked = val[:4] + "…****" if len(val) > 4 and "NOT SET" not in val else val
        print(f"    {env_var:<20}  {masked:<20}  {source}")
    print()
    print("  Intent flags:")
    print(f"    eps={args.eps}  min_samples={args.min_samples}  "
          f"algorithm={args.clustering_algorithm}  "
          f"dedup={args.dedup_method}  "
          f"min_token={args.min_token_length}")
    print(f"    max_clusters={args.max_clusters}  "
          f"representatives={args.representatives}  "
          f"keywords={args.keywords}")
    if args.use_umap:
        print(f"    umap=ON  components={args.umap_components}  "
              f"neighbors={args.umap_neighbors}  "
              f"min_dist={args.umap_min_dist}")
    else:
        print( "    umap=OFF")
    print()
    print("  FP flags:")
    print(f"    max_categories={args.max_categories}  "
          f"samples_per_category={args.samples_per_category}  "
          f"min_occurrences={args.min_occurrences}")
    print("=" * 66 + "\n")


# Sleep helper — interruptible in 60-second chunks

def _interruptible_sleep(seconds: float):
    """Sleep for `seconds` total, but wake every 60 s to check _SHUTDOWN."""
    deadline = time.time() + seconds
    while not _SHUTDOWN:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(60, remaining))

# Main loop

def main():
    parser = build_parser()
    args   = parser.parse_args()

    setup_logging(args.debug, args.log_file)

    if args.dry_run:
        print_dry_run(args)
        sys.exit(0)

    config     = build_config(args)
    engine     = DLPAnalysisEngine(config)
    output_dir = Path(args.output_dir)
    state_file = Path(args.state_file)
    interval_s = args.interval_hours * 3600

    log.info(
        f"DLP Scheduler starting — "
        f"mode={args.mode}  "
        f"interval={args.interval_hours}h  "
        f"lookback={args.lookback_hours}h  "
        f"output={output_dir}/  "
        f"state={'disabled' if args.no_state else state_file}"
    )
    if args.run_once:
        log.info("--run-once: will exit after the first run.")

    run_number = 0

    while not _SHUTDOWN:
        #  Check whether a run is actually due 
        if not args.no_state:
            secs_left = _seconds_until_next_run(state_file, interval_s)
            if secs_left > 0 and run_number > 0:
                # Only skip on subsequent loops; always run at least once
                h, rem = divmod(int(secs_left), 3600)
                m      = rem // 60
                log.info(
                    f"Next run not due for {h}h {m}m — sleeping. "
                    f"(Send SIGTERM / Ctrl-C to stop.)"
                )
                _interruptible_sleep(secs_left)
                continue
        elif run_number == 0 and not args.no_state:
            # First ever start with a fresh state file — run immediately
            pass

        if _SHUTDOWN:
            break

        #  Execute one analysis run 
        run_number += 1
        log.info(f"Run #{run_number} — due now.")

        success = run_analysis(args, engine, output_dir)

        if success and not args.no_state:
            _write_state(state_file, time.time())
            log.info(f"State file updated → {state_file}")
        elif not success:
            log.warning(
                f"Run #{run_number} finished with errors. "
                f"State file NOT updated — next run will retry immediately."
            )

        if args.run_once or _SHUTDOWN:
            break

        #  Schedule next run 
        next_dt = datetime.utcfromtimestamp(time.time() + interval_s).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
        log.info(
            f"Run #{run_number} complete. "
            f"Next run at {next_dt} ({args.interval_hours}h from now). "
            f"Send SIGTERM or Ctrl-C to stop cleanly."
        )
        _interruptible_sleep(interval_s)

    log.info("DLP Scheduler stopped.")


if __name__ == "__main__":
    main()