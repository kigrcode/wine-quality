import pandas as pd
import numpy as np

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
