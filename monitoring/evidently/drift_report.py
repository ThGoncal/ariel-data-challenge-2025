"""Drift monitoring with Evidently."""
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import DatasetDriftMetric
import yaml


def generate_drift_report(reference_path: str, current_path: str):
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        DatasetDriftMetric(),
    ])

    report.run(reference_data=reference, current_data=current)
    report.save_html("monitoring/evidently/reports/drift_report.html")

    results = report.as_dict()
    drift_detected = results["metrics"][2]["result"]["dataset_drift"]

    print(f"{'⚠️  Drift détecté !' if drift_detected else '✅ Pas de drift détecté'}")
    return drift_detected


if __name__ == "__main__":
    generate_drift_report(
        reference_path="data/processed/reference.csv",
        current_path="data/processed/current.csv"
    )
