"""Feature engineering pipeline."""
import pandas as pd
from loguru import logger


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme les données brutes en features.
    Args:
        df: DataFrame brut
    Returns:
        DataFrame avec les features engineerées
    """
    logger.info(f"Building features for {len(df)} rows")
    # TODO: implémenter vos transformations
    return df
