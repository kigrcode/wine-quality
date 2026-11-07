# Wine Quality Classification

![CI](https://github.com/kigrcode/wine-quality/actions/workflows/ci.yml/badge.svg)

## Project Summary
Classification task predicting whether a red wine is rated "good" (quality 6 or above), with a test ROC-AUC of about 0.83. The data set is the UCI Wine Quality (red) data: 1,599 rows, reduced to 1,359 after removing duplicates. "Good" is defined as any wine with a quality score of 6 or more.

Errors concentrate around the borderline scores of 5 and 6. Tuning helped XGBoost but not logistic regression, and neither threshold tuning nor feature engineering improved performance.

### Results (ROC-AUC)

| Stage | ROC-AUC | Note |
|---|---|---|
| XGBoost defaults (CV) | 0.813 | like-for-like baseline |
| Baseline with early stopping (CV) | 0.838 | flattered: early stopping sees the validation fold |
| Tuned (CV) | 0.830 | best of 50 Optuna trials, slightly optimistic |
| Final test | 0.825 | 95% bootstrap interval 0.776 to 0.873 (about 270 test wines) |

### Key findings
- **Tuning:** raised XGBoost from 0.813 to 0.830 in cross-validation. For logistic regression the gain (+0.003) was within noise.
- **Feature engineering:** interactions, polynomial terms and binning did not help tree models on the 11 raw features, and stacking them all overfitted.
- **Threshold tuning:** a threshold of 0.40 gained +0.010 F1 on out-of-fold training predictions, but lost 0.010 on the test set. The F1 curve is flat between 0.35 and 0.50, so there is no reliable benefit.
- **Error analysis:** at the tuned threshold, 282 of 1,087 training wines (26%) are misclassified. Quality 5 and 6 account for 93% of the errors, and all 40 confident errors are one score either side of the cut. The model's probabilities are well calibrated: the error rate in each band roughly matches its stated probability.
- **Ceiling:** the labels are human taster scores cut at 6, so a 5 and a 6 may be nearly the same wine. A possible next step is to predict the quality score itself and apply the cut afterwards, keeping the information the yes/no label discards.

## About the workflow
An end-to-end classical machine learning project, built from my reusable [`ml-workflow-template`](https://github.com/kigrcode/ml-workflow-template). It separates orchestration notebooks from reusable `src` modules, with all dataset-specific settings centralised in `config.yaml`.

## Workflow
Run notebooks in order:
1. `1. EDA`: exploratory data analysis
2. `2. Data Preprocessing`: feature type detection, imputation, encoding, scaling
3. `3. Baseline Models`: cross-validated comparison of all models
4. `4. Feature Engineering`: automated and domain-specific feature generation
5. `5. Hyperparameter Tuning`: Optuna Bayesian optimisation or random search
6. `6. Threshold Tuning`: optimal classification threshold selection
7. `7. Final Evaluation`: single test set evaluation, reports, artifacts and MLflow logging
8. `8. Error Analysis`: errors by confidence band and by a grouping column, for classification

## Project structure

```
wine-quality/
├── .github/workflows/    # CI: runs the tests on every pull request
├── .gitignore            # keeps data, artifacts, models and MLflow files out of git
├── data/
│   ├── raw/              # original immutable data (gitignored; see "Add your data")
│   ├── interim/          # engineered features pre-preprocessing
│   └── processed/        # final preprocessed data for modelling
├── artifacts/            # saved trainers, preprocessor, tuned models, threshold (gitignored)
├── models/
│   ├── baseline/         # all baseline models
│   ├── tuned/            # hyperparameter tuned models
│   └── final/            # definitive model for deployment
├── notebooks/            # one notebook per workflow stage (1 to 8)
├── reports/              # evaluation reports, error analysis and figures
├── src/
│   ├── analysis/         # error-analysis helpers
│   ├── config/           # paths, settings, config loader
│   ├── data/             # data loading utilities
│   ├── features/         # pipeline, type detection, feature engineering
│   ├── models/
│   │   ├── training/     # trainer, registry, metrics, tuning, threshold
│   │   └── prediction/   # prediction utilities
│   ├── tracking/         # MLflow logging helper
│   ├── utils/            # reporting utilities
│   └── visualization/    # EDA and evaluation plotting functions
├── tests/                # unit tests
├── config.yaml           # all dataset-specific settings
├── environment.yml       # conda environment
├── mlflow.db             # MLflow run history (gitignored; created on first run)
└── pyproject.toml        # makes src installable (pip install -e .)
```

## Setup

### 1. Create and activate the environment
```bash
conda env create -f environment.yml
conda activate wine-quality
```

### 2. Install the project as a package
```bash
pip install -e .
```
This makes `src` modules importable from anywhere in the project.

### 3. Add your data
The `data/` folder is gitignored, so the data has to be rebuilt:

1. Download the red wine file (`winequality-red.csv`) from the "Wine Quality" data set in the UCI Machine Learning Repository. The original is semicolon-separated.
2. Convert it to a comma-separated file with a `good` column, and save it as `data/raw/winequality-red.csv`:

```python
import pandas as pd

df = pd.read_csv("winequality-red.csv", sep=";")
df["good"] = (df["quality"] >= 6).astype(int)
df.to_csv("data/raw/winequality-red.csv", index=False)
```

Raw data should otherwise be unmodified: missing values, duplicates and transformations are handled by the workflow.

### 4. Configure the project
`config.yaml` is the only file that needs to change for a new project. The key settings for this one:

```yaml
target: "good"
drop_columns:
  - "quality"          # the score the target was built from; excluded from the features
drop_dupes: true
n_jobs: 4              # caps threads used by XGBoost and LightGBM
error_analysis:
  group_column: quality
```

### 5. Run the tests and open MLflow
```bash
pytest
mlflow ui              # run from the folder containing mlflow.db
```

## Design decisions

### Preprocessing inside CV folds
The preprocessor is fitted inside each cross validation fold on training data only, then applied to the validation fold to prevent data leakage.

### Test set held out until final evaluation
The test set is split off at the start of the preprocessing notebook and never touched until notebook 7. All model selection, feature engineering and tuning decisions are made using CV scores only. Threshold tuning and error analysis use out-of-fold predictions on the training data.

### Config-driven design
All dataset-specific settings live in `config.yaml`: target column, columns to drop, split parameters, pipeline settings, model selection and tuning parameters. Notebooks and `src` modules contain no hardcoded dataset-specific values. Starting a new project means editing `config.yaml` only.

### Thread cap
XGBoost and LightGBM use every core by default, which slows small cross-validation runs badly. The `n_jobs` setting in `config.yaml` caps the threads in both the trainer and the tuning code.

### Feature engineering after baseline modelling
Feature engineering is deliberately placed after baseline modelling. SHAP values from the baseline model inform which features are worth engineering.

### Two-stage feature selection
Feature selection uses mutual information as a fast pre-filter to eliminate obvious noise, followed by SHAP-based refinement for precise, model-aware selection.

### Automated feature engineering is opt-in
All automated feature engineering (interactions, polynomial, binning, aggregations) is disabled by default in `config.yaml`.

### Optional dependencies
XGBoost, LightGBM and CatBoost are imported conditionally. If any are not installed the model registry still loads and the workflow runs with whatever is available.

## Usage notes

### Threshold tuning metric
Choose the threshold metric in `config.yaml` based on the business problem:
- `f1`: balanced precision and recall, good default
- `recall`: minimise false negatives, use for medical or fraud detection
- `precision`: minimise false positives, use for spam or ad targeting
- `accuracy`: use only when classes are balanced

### Feature engineering flags
Enable selectively based on model type and dataset size:
- `interactions` and `polynomial`: most useful for linear models
- `binning`: useful for non-monotonic numeric relationships
- `aggregations`: useful when group statistics carry signal

## Development notes
AI-assisted development workflow was used throughout, with architectural decisions, methodological choices and code review handled by the developer.


