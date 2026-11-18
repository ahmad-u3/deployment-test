import numpy as np
import pandas as pd

DIST_COLS = ['dist1', 'dist2', 'dist3', 'dist4']
DROP_COLS = DIST_COLS + ['has_bus_terminal']


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering as Section 9, applied to raw input."""
    df = df.copy()
    df = df.where(df.notna(), np.nan)          # None (from JSON) -> NaN
    df['avg_dist'] = df[DIST_COLS].astype(float).mean(axis=1)
    return df.drop(columns=DROP_COLS, errors='ignore')
