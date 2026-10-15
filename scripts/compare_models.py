"""Print the recorded baseline-versus-fine-tuned metric comparison."""
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "results" / "metrics" / "comparison.csv"


def main():
    with METRICS_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"metric", "baseline", "fine_tuned"}.issubset(rows[0]):
        raise ValueError(f"Metric file is missing expected columns: {METRICS_PATH}")

    label_width = max(len("Metric"), *(len(row["metric"]) for row in rows))
    print(f"{'Metric':<{label_width}}  {'Baseline':>10}  {'Fine-tuned':>10}  {'Change':>9}")
    print(f"{'-' * label_width}  {'-' * 10}  {'-' * 10}  {'-' * 9}")
    for row in rows:
        baseline = float(row["baseline"])
        fine_tuned = float(row["fine_tuned"])
        print(f"{row['metric']:<{label_width}}  {baseline:>10.3f}  {fine_tuned:>10.3f}  {fine_tuned-baseline:>+9.3f}")
    print("\nValues are project-reported, not newly evaluated by this script.")


if __name__ == "__main__":
    main()
