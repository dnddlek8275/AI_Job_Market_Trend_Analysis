# AI Job Market Trend Analysis

A machine learning project for analyzing salary-related patterns in AI job market data.

## Overview

The repository contains experiments for predicting the `high_salary` target using multiple classification models:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Model experiments are available as Jupyter notebooks. The `train_salary_models_with_mlflow.py` script trains and evaluates the models, logs metrics and artifacts with MLflow, and supports model registration.

## Project Structure

- `data/`: Dataset files managed with DVC
- `*_salary_random_split.ipynb`: Individual model experiments
- `logistic_regression_salary_cross_validation.ipynb`: Cross-validation experiment
- `train_salary_models_with_mlflow.py`: MLflow-based training and evaluation pipeline
- `.dvc/`, `.dvcignore`: DVC configuration

## Data

The training script expects:

```text
data/salary_prediction_monthly_dataset.csv
```

If the dataset is managed through DVC, restore it before training:

```bash
dvc pull
```

## Run the Training Pipeline

Install the packages used by the project, including pandas, scikit-learn, MLflow, XGBoost, LightGBM, and CatBoost. Then run:

```bash
python train_salary_models_with_mlflow.py
```
