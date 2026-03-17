"""Monitoring du drift avec Evidently."""
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestShareOfDriftedColumns
from pathlib import Path
from loguru import logger
from datetime import datetime


def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_dir: str = "monitoring/evidently/reports"
) -> dict:
    """Génère un rapport de drift Evidently."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    report_path = f"{output_dir}/drift_report_{timestamp}.html"
    report.save_html(report_path)
    logger.info(f"Rapport Evidently sauvegardé : {report_path}")

    tests = TestSuite(tests=[TestShareOfDriftedColumns(lt=0.3)])
    tests.run(reference_data=reference_data, current_data=current_data)
    test_results = tests.as_dict()

    return {
        "report_path": report_path,
        "tests_passed": test_results["summary"]["all_passed"],
        "timestamp": timestamp,
    }
