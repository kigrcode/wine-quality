"""
Feature engineering utilities.
Extend this per project.
"""

import pandas as pd


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add project-specific features.
    Checks key wine quality ratios / combinations.
    """
    df = df.copy()
    df["free_so2_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 0.1)
    df["total_acidity"] = df["fixed acidity"] + (df["volatile acidity"])
    df["density_to_alcohol"] = df["density"] / (df["alcohol"] + 0.1)
    return df