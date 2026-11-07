import mlflow
import pytest

from src.tracking.mlflow_logging import log_final_run


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment(
        "test-exp", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    return tmp_path


def test_logs_params_and_metrics(tracking):
    run_id = log_final_run("test-exp", "run-1", {"lr": 0.1, "depth": 3}, {"auc": 0.82}, [])
    run = mlflow.get_run(run_id)
    assert run.data.params == {"lr": "0.1", "depth": "3"}  # MLflow stores params as text
    assert run.data.metrics["auc"] == pytest.approx(0.82)


def test_skips_none_metrics(tracking):
    run_id = log_final_run("test-exp", "run-2", {}, {"auc": 0.8, "missing": None}, [])
    metrics = mlflow.get_run(run_id).data.metrics
    assert "auc" in metrics
    assert "missing" not in metrics


def test_logs_artifact_files(tracking):
    report = tracking / "report.json"
    report.write_text("{}")
    run_id = log_final_run("test-exp", "run-3", {}, {}, [report])
    names = [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)]
    assert "report.json" in names


def test_run_is_finished_and_closed(tracking):
    run_id = log_final_run("test-exp", "run-4", {}, {}, [])
    assert mlflow.get_run(run_id).info.status == "FINISHED"
    assert mlflow.active_run() is None