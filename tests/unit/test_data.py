"""Unit tests for data ingestion."""
import pytest
import pandas as pd
from pathlib import Path


def test_load_raw_data_csv(tmp_path):
    """Test loading CSV data."""
    # Create a temp CSV
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    
    from src.data.ingest import load_raw_data
    result = load_raw_data(csv_path)
    
    assert len(result) == 2
    assert list(result.columns) == ["a", "b"]
