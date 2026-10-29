import pandas as pd

from src.data.load_data import prepare_data


def _make_df():
    return pd.DataFrame({
        "a": [1, 1, 2, 3],
        "b": [10, 10, 20, 30],
        "target": [0, 0, 1, 1],
    })


def test_drop_dupes_removes_duplicate_rows():
    config = {"target": "target", "drop_dupes": True}
    X, y = prepare_data(_make_df(), config)
    assert len(X) == 3
    assert len(y) == 3
    assert X.index.tolist() == [0, 1, 2]


def test_drop_dupes_is_off_by_default():
    config = {"target": "target"}
    X, y = prepare_data(_make_df(), config)
    assert len(X) == 4


def test_drop_dupes_does_not_modify_input_frame():
    df = _make_df()
    prepare_data(df, {"target": "target", "drop_dupes": True})
    assert len(df) == 4