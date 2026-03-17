"""Unit tests for training module."""
import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from unittest.mock import patch


def test_train_model_returns_run_id():
    """train_model should return a valid MLflow run_id string."""
    from src.models.train import train_model

    df = pd.DataFrame({"feature_1": [1, 2, 3, 4, 5], "target": [0, 1, 0, 1, 0]})
    model = LogisticRegression()

    with patch("mlflow.start_run") as mock_run, \
         patch("mlflow.sklearn.log_model"), \
         patch("mlflow.log_params"), \
         patch("mlflow.log_metrics"):
        mock_run.return_value.__enter__.return_value.info.run_id = "test-run-id"
        run_id = train_model(df, "target", model, experiment_name="test")
        assert isinstance(run_id, str)
