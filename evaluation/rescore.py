"""Offline rescore of a previously saved baseline; never invokes the Agent or APIs."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.failure_analysis import build_failure_analysis
from evaluation.metrics import aggregate_results, score_case
from evaluation.relevance_analysis import build_relevance_analysis
from evaluation.reporting import print_summary


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "evaluation" / "results"


def rescore_report(report: dict, dataset: dict) -> dict:
    """Recompute scores from saved actual outputs and current deterministic rules."""
    ground_truth = {case["id"]: case for case in dataset["cases"]}
    rescored_cases = []
    for saved in report.get("cases", []):
        case = ground_truth.get(saved.get("id"))
        if case is None:
            raise ValueError(f"Case {saved.get('id')} is missing from the dataset")
        row = dict(saved)
        row["query"] = case["turns"][-1]
        row["expected"] = case["expected"]
        row["scores"] = score_case(case, row.get("actual", {}))
        rescored_cases.append(row)
    return {
        **report,
        "rescored_at": datetime.now(timezone.utc).isoformat(),
        "summary": aggregate_results(rescored_cases),
        "cases": rescored_cases,
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline rescore of saved Evaluation outputs.")
    parser.add_argument("--input", type=Path, default=RESULTS_DIR / "latest.json")
    parser.add_argument("--dataset", type=Path, default=PROJECT_DIR / "evaluation/datasets/baseline_v1.json")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "rescored.json")
    parser.add_argument("--failure-output", type=Path, default=RESULTS_DIR / "failure_analysis.json")
    parser.add_argument("--relevance-output", type=Path, default=RESULTS_DIR / "relevance_analysis.json")
    args = parser.parse_args()

    report = _load_json(args.input)
    rescored = rescore_report(report, _load_json(args.dataset))
    failures = build_failure_analysis(rescored)
    relevance = build_relevance_analysis(rescored["cases"])
    _write_json(args.output, rescored)
    _write_json(args.failure_output, failures)
    _write_json(args.relevance_output, relevance)

    print_summary(rescored["summary"])
    print(f"\nRescored report: {args.output}")
    print(f"Failure analysis: {args.failure_output} ({failures['failure_count']} cases)")
    print(f"Relevance analysis: {args.relevance_output}")


if __name__ == "__main__":
    main()
