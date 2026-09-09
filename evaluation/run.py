"""Command-line entry point for the baseline evaluation harness."""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from evaluation.harness import build_production_harness, load_dataset
from evaluation.failure_analysis import build_failure_analysis
from evaluation.relevance_analysis import build_relevance_analysis
from evaluation.reporting import print_summary


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_DIR / "evaluation" / "datasets" / "baseline_v1.json"
DEFAULT_OUTPUT = PROJECT_DIR / "evaluation" / "results" / "latest.json"
DEFAULT_FAILURE_OUTPUT = PROJECT_DIR / "evaluation" / "results" / "failure_analysis.json"
DEFAULT_RELEVANCE_OUTPUT = PROJECT_DIR / "evaluation" / "results" / "relevance_analysis.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic baseline evaluation against the current Agent.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failure-output", type=Path, default=DEFAULT_FAILURE_OUTPUT)
    parser.add_argument("--relevance-output", type=Path, default=DEFAULT_RELEVANCE_OUTPUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dataset = load_dataset(args.dataset)
    harness = build_production_harness(PROJECT_DIR)
    report = harness.run(dataset)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    failure_analysis = build_failure_analysis(report)
    args.failure_output.parent.mkdir(parents=True, exist_ok=True)
    args.failure_output.write_text(
        json.dumps(failure_analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    relevance_analysis = build_relevance_analysis(report["cases"])
    args.relevance_output.parent.mkdir(parents=True, exist_ok=True)
    args.relevance_output.write_text(
        json.dumps(relevance_analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_summary(report["summary"])
    print(f"\nMachine-readable report: {args.output}")
    print(f"Failure analysis: {args.failure_output} ({failure_analysis['failure_count']} cases)")
    print(f"Relevance analysis: {args.relevance_output}")


if __name__ == "__main__":
    main()
