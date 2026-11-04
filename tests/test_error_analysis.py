import numpy as np
import pandas as pd
import pytest

from src.analysis.error_analysis import build_error_table, error_rate_by_bands
from src.analysis.error_analysis import build_error_table, error_rate_by_bands, confident_errors


@pytest.fixture
def small_table():
    # predictions at 0.5: [0, 0, 0, 1, 1, 1]; only the last wine is wrong
    y_true = [0, 0, 0, 1, 1, 0]
    y_proba = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    return build_error_table(y_true, y_proba, threshold=0.5)

@pytest.fixture
def mixed_table():
    # predictions at 0.5: [1, 1, 1, 0, 0, 0]; wrong rows are 1 (p=0.9) and 4 (p=0.1)
    y_true = [1, 0, 1, 0, 1, 0]
    y_proba = [0.95, 0.9, 0.5, 0.05, 0.1, 0.3]
    return build_error_table(y_true, y_proba, threshold=0.5)


def test_build_error_table_columns_and_flags():
    table = build_error_table([1, 0, 1, 0], [0.9, 0.6, 0.3, 0.1], 0.5)
    assert list(table.columns) == ["true", "proba", "pred", "wrong"]
    assert table["pred"].tolist() == [1, 1, 0, 0]
    assert table["wrong"].tolist() == [False, True, True, False]


def test_threshold_is_respected():
    table = build_error_table([1, 0, 1, 0], [0.9, 0.6, 0.3, 0.1], 0.7)
    assert table["pred"].tolist() == [1, 0, 0, 0]


def test_probability_equal_to_threshold_is_positive():
    table = build_error_table([1], [0.5], 0.5)
    assert table["pred"].tolist() == [1]


def test_accepts_series_with_nonstandard_index():
    y_true = pd.Series([1, 0, 1], index=[10, 20, 30])
    table = build_error_table(y_true, np.array([0.9, 0.1, 0.8]), 0.5)
    assert table["wrong"].sum() == 0
    assert list(table.index) == [0, 1, 2]


def test_error_rate_by_bands_values(small_table):
    result = error_rate_by_bands(small_table, bins=[0, 0.5, 1.0])
    assert result["n"].tolist() == [3, 3]
    assert result["error_rate"].tolist() == pytest.approx([0.0, 1 / 3])


def test_error_rate_by_bands_keeps_empty_bands(small_table):
    result = error_rate_by_bands(small_table, bins=[0, 0.2, 0.25, 1.0])
    assert result["n"].tolist() == [2, 0, 4]


def test_error_rate_by_bands_does_not_modify_input(small_table):
    before = small_table.copy()
    error_rate_by_bands(small_table, bins=[0, 0.5, 1.0])
    pd.testing.assert_frame_equal(small_table, before)


def test_probability_of_exactly_zero_is_included():
    table = build_error_table([0, 1], [0.0, 0.9], 0.5)
    result = error_rate_by_bands(table, bins=[0, 0.5, 1.0])
    assert result["n"].sum() == 2





def test_confident_errors_returns_only_confident_wrong_rows(mixed_table):
    result = confident_errors(mixed_table, margin=0.2)
    assert result.index.tolist() == [1, 4]
    assert result["wrong"].all()


def test_confident_errors_excludes_hesitant_wrong_rows():
    table = build_error_table([1, 0], [0.4, 0.6], threshold=0.5)  # both wrong, near 0.5
    result = confident_errors(table, margin=0.2)
    assert len(result) == 0


def test_confident_errors_empty_result_keeps_columns(mixed_table):
    result = confident_errors(mixed_table, margin=0.01)
    assert len(result) == 0
    assert list(result.columns) == list(mixed_table.columns)


def test_confident_errors_does_not_modify_input(mixed_table):
    before = mixed_table.copy()
    confident_errors(mixed_table, margin=0.2)
    pd.testing.assert_frame_equal(mixed_table, before)