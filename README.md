# AI Job Market Trend Analysis

A reproducible machine learning experiment for predicting whether an AI job belongs to the high-salary class.

The project compares five binary classifiers, tracks every training run with MLflow, stores the dataset through DVC, and can register the best model by ROC AUC in the MLflow Model Registry.

## Experiment Workflow

1. Restore the dataset managed by DVC.
2. Load `data/salary_prediction_monthly_dataset.csv`.
3. Keep numeric columns, remove rows with missing values, and use `high_salary` as the target.
4. Create a stratified 80/20 train-test split with random seed `42`.
5. Train five classification models.
6. Log parameters, metrics, and model artifacts to MLflow.
7. Select the run with the highest ROC AUC.
8. Register that model as `salary_classifier` with the `production` alias.

## Models

| Model | Key Configuration |
| --- | --- |
| Logistic Regression | StandardScaler pipeline, LBFGS solver, 1,000 maximum iterations |
| Random Forest | 350 trees, balanced subsampling, minimum leaf size of 2 |
| XGBoost | 450 trees, histogram tree method, class-imbalance weighting |
| LightGBM | 450 estimators, 31 leaves, balanced class weights |
| CatBoost | 500 iterations, depth 5, automatically balanced class weights |

## Evaluation Metrics

Every MLflow run records:

- Accuracy
- Precision
- Recall
- F1 score
- ROC AUC
- PR AUC
- Log loss

## Repository Structure

```text
.
├── .dvc/
├── data/
│   └── salary_prediction_monthly_dataset.csv.dvc
├── catboost_salary_random_split.ipynb
├── lightgbm_salary_random_split.ipynb
├── logistic_regression_salary_cross_validation.ipynb
├── random_forest_salary_random_split.ipynb
├── xgboost_salary_random_split.ipynb
└── train_salary_models_with_mlflow.py
```

The notebooks contain individual model experiments. `train_salary_models_with_mlflow.py` provides the consolidated MLflow training and model-registration workflow.

## Requirements

The training script imports the following packages:

- pandas
- scikit-learn
- MLflow
- XGBoost
- LightGBM
- CatBoost

DVC is also required to restore the tracked dataset.

## Restore the Data

```bash
dvc pull
```

The training script expects the restored file at:

```text
data/salary_prediction_monthly_dataset.csv
```

## MLflow Configuration

By default, the script connects to:

```text
http://localhost:5000
```

Set `MLFLOW_TRACKING_URI` to use a different tracking server.

## Run the Experiment

Train all models and register the best run:

```bash
python train_salary_models_with_mlflow.py
```

Train and log the runs without registering a model:

```bash
python train_salary_models_with_mlflow.py --skip-register
```

## Current Scope

This repository focuses on model comparison and experiment tracking. It does not currently include a packaged inference service, automated deployment pipeline, or application UI.
