"""Data drift monitoring with Evidently."""
from pathlib import Path
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from loguru import logger


def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: str | Path = "monitoring/evidently/reports/drift_report.html",
) -> None:
    """Generate a data drift report using Evidently."""
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output))
    logger.info(f"Drift report saved to {output}")
