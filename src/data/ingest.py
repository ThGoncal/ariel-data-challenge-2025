"""Data ingestion module."""
import pandas as pd
from pathlib import Path
from loguru import logger


def load_raw_data(source_path: str | Path) -> pd.DataFrame:
    """Load raw data from source."""
    path = Path(source_path)
    logger.info(f"Loading data from {path}")
    loaders = {
        ".csv": pd.read_csv,
        ".parquet": pd.read_parquet,
        ".json": pd.read_json,
    }
    loader = loaders.get(path.suffix)
    if not loader:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    df = loader(path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save processed data."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Saved processed data to {path}")
