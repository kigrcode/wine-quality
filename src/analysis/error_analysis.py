import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.load_data import prepare_data

def build_error_table(y_true, y_proba, threshold):
    """Return a DataFrame with columns: true, proba, pred, wrong."""

    df_error = pd.DataFrame({
    "true": np.asarray(y_true),
    "proba": np.asarray(y_proba),
    })

    df_error["pred"] = (df_error["proba"] >= threshold).astype(int) 
    df_error["wrong"] = df_error["true"] != df_error["pred"]

    return df_error

def error_rate_by_bands(table, bins):
    """Return the error rate and wine count for each probability band."""

    table = table.copy()
    table["band"] = pd.cut(table["proba"], bins=bins, include_lowest=True)  

    result = table.groupby("band", observed=False)["wrong"].agg(["mean", "count"])
    result = result.rename(columns={"mean": "error_rate", "count": "n"})
    return result

def confident_errors(table, margin):
    """Return the wrong rows whose probability is within `margin` of 0 or 1."""
    mask = table["wrong"] & ((table["proba"] > 1- margin)    | (table["proba"] < margin))
    return table[mask].copy()

def aligned_column(df, config, column):
    """Return `column` for the training rows, in the same order as X_train.

    Reproduces the pipeline's duplicate drop and train/test split on the raw
    data, so the returned Series lines up row for row with the processed
    training set.
    """
    SPLIT_CONFIG = config["split"]

    if config.get("drop_dupes", False):
       df = df.drop_duplicates().reset_index(drop=True)
    col = df[column]

    X, y = prepare_data(df, config)
    stratify_col = y if SPLIT_CONFIG["stratify"] else None

    _, _, _, _, col_tr, _ = train_test_split(
    X, y, col,
    test_size=SPLIT_CONFIG["test_size"],
    random_state=SPLIT_CONFIG["random_state"],
    stratify=stratify_col,
    )
    return col_tr