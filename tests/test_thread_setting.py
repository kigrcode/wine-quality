import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

from src.models.training.model_registry import apply_thread_setting


def test_none_returns_params_unchanged():
    params = {"n_estimators": 10}
    assert apply_thread_setting(params, RandomForestClassifier, None) == params


def test_xgboost_gets_n_jobs():
    xgboost = pytest.importorskip("xgboost")
    out = apply_thread_setting({"n_jobs": -1}, xgboost.XGBClassifier, 4)
    assert out["n_jobs"] == 4


def test_lightgbm_gets_n_jobs_even_without_default():
    lightgbm = pytest.importorskip("lightgbm")
    out = apply_thread_setting({"n_estimators": 10}, lightgbm.LGBMClassifier, 4)
    assert out["n_jobs"] == 4


def test_catboost_gets_thread_count_not_n_jobs():
    catboost = pytest.importorskip("catboost")
    out = apply_thread_setting({"iterations": 10}, catboost.CatBoostClassifier, 4)
    assert out["thread_count"] == 4
    assert "n_jobs" not in out


def test_random_forest_overrides_existing_n_jobs():
    out = apply_thread_setting({"n_jobs": -1}, RandomForestClassifier, 4)
    assert out["n_jobs"] == 4


def test_model_without_thread_option_is_left_alone():
    out = apply_thread_setting({"alpha": 1.0}, Ridge, 4)
    assert "n_jobs" not in out
    assert "thread_count" not in out


def test_input_dict_is_not_modified():
    params = {"n_jobs": -1}
    apply_thread_setting(params, RandomForestClassifier, 4)
    assert params == {"n_jobs": -1}
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

from src.models.training.model_registry import apply_thread_setting


def test_none_returns_params_unchanged():
    params = {"n_estimators": 10}
    assert apply_thread_setting(params, RandomForestClassifier, None) == params


def test_xgboost_gets_n_jobs():
    xgboost = pytest.importorskip("xgboost")
    out = apply_thread_setting({"n_jobs": -1}, xgboost.XGBClassifier, 4)
    assert out["n_jobs"] == 4


def test_lightgbm_gets_n_jobs_even_without_default():
    lightgbm = pytest.importorskip("lightgbm")
    out = apply_thread_setting({"n_estimators": 10}, lightgbm.LGBMClassifier, 4)
    assert out["n_jobs"] == 4


def test_catboost_gets_thread_count_not_n_jobs():
    catboost = pytest.importorskip("catboost")
    out = apply_thread_setting({"iterations": 10}, catboost.CatBoostClassifier, 4)
    assert out["thread_count"] == 4
    assert "n_jobs" not in out


def test_random_forest_overrides_existing_n_jobs():
    out = apply_thread_setting({"n_jobs": -1}, RandomForestClassifier, 4)
    assert out["n_jobs"] == 4


def test_model_without_thread_option_is_left_alone():
    out = apply_thread_setting({"alpha": 1.0}, Ridge, 4)
    assert "n_jobs" not in out
    assert "thread_count" not in out


def test_input_dict_is_not_modified():
    params = {"n_jobs": -1}
    apply_thread_setting(params, RandomForestClassifier, 4)
    assert params == {"n_jobs": -1}