import mlflow


def log_final_run(experiment_name, run_name, params, metrics, artifact_paths):
    """Log one run's parameters, metrics and files to MLflow; return the run id."""
  
    mlflow.set_experiment(experiment_name)

    clean_metrics = {k: v for k, v in metrics.items() if v is not None}

    with mlflow.start_run(run_name=run_name) as run:

        mlflow.log_params(params)
      
        mlflow.log_metrics(clean_metrics)
     
        for path in artifact_paths:
            mlflow.log_artifact(str(path))

    return run.info.run_id