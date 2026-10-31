import pandas as pd

from src.features.build_features import add_domain_features


def _wine_df():
    return pd.DataFrame({
        "fixed acidity": [7.0, 8.0],
        "volatile acidity": [0.5, 0.25],
        "free sulfur dioxide": [10.0, 20.0],
        "total sulfur dioxide": [40.0, 50.0],
        "density": [1.0, 0.99],
        "alcohol": [10.0, 9.9],
    })


def test_adds_expected_columns():
    out = add_domain_features(_wine_df())
    for col in ["free_so2_ratio", "total_acidity", "density_to_alcohol"]:
        assert col in out.columns
    assert out.shape[1] == 9


def test_feature_values():
    out = add_domain_features(_wine_df())
    assert out["total_acidity"].tolist() == [7.5, 8.25]
    assert abs(out["free_so2_ratio"].iloc[0] - 10 / 40.1) < 1e-9


def test_does_not_modify_input_frame():
    df = _wine_df()
    columns_before = list(df.columns)
    add_domain_features(df)
    assert list(df.columns) == columns_before


def test_zero_total_so2_gives_finite_ratio():
    df = _wine_df()
    df.loc[0, "total sulfur dioxide"] = 0.0
    out = add_domain_features(df)
    assert out["free_so2_ratio"].notna().all()
    assert (out["free_so2_ratio"] != float("inf")).all()