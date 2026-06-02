import argparse
import importlib.util
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/salary_prediction_monthly_dataset.csv")
TARGET_COL = "high_salary"
RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPERIMENT_NAME = "salary_prediction_models"
REGISTERED_MODEL_NAME = "salary_classifier"


def require_package(module_name, package_name=None):
    if importlib.util.find_spec(module_name) is None:
        package = package_name or module_name
        raise ModuleNotFoundError(
            f"{package} is not installed. Install it first, then run this script again."
        )


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. If the file is managed by DVC, run `dvc pull` first."
        )

    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} column not found in {DATA_PATH}.")

    df = df.select_dtypes(include=["number"]).dropna()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return X, y


def build_models(y_train):
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count

    models = {
        "LogisticRegression": {
            "model": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=RANDOM_STATE,
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
            "params": {
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
                "solver": "lbfgs",
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                n_estimators=350,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "params": {
                "n_estimators": 350,
                "max_depth": "None",
                "min_samples_leaf": 2,
                "class_weight": "balanced_subsample",
                "random_state": RANDOM_STATE,
                "n_jobs": 1,
            },
        },
    }

    require_package("xgboost")
    from xgboost import XGBClassifier

    models["XGBoost"] = {
        "model": XGBClassifier(
            n_estimators=450,
            learning_rate=0.04,
            max_depth=4,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric=["logloss", "auc"],
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        ),
        "params": {
            "n_estimators": 450,
            "learning_rate": 0.04,
            "max_depth": 4,
            "min_child_weight": 2,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "binary:logistic",
            "scale_pos_weight": scale_pos_weight,
            "random_state": RANDOM_STATE,
            "tree_method": "hist",
        },
    }

    require_package("lightgbm")
    from lightgbm import LGBMClassifier

    models["LightGBM"] = {
        "model": LGBMClassifier(
            n_estimators=450,
            learning_rate=0.04,
            max_depth=-1,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "params": {
            "n_estimators": 450,
            "learning_rate": 0.04,
            "max_depth": -1,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        },
    }

    require_package("catboost")
    from catboost import CatBoostClassifier

    models["CatBoost"] = {
        "model": CatBoostClassifier(
            iterations=500,
            learning_rate=0.04,
            depth=5,
            loss_function="Logloss",
            eval_metric="AUC",
            custom_metric=["Logloss", "Accuracy", "Precision", "Recall", "F1"],
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            allow_writing_files=False,
            thread_count=-1,
            verbose=False,
        ),
        "params": {
            "iterations": 500,
            "learning_rate": 0.04,
            "depth": 5,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "auto_class_weights": "Balanced",
            "random_seed": RANDOM_STATE,
        },
    }

    return models


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "pr_auc": average_precision_score(y_test, y_pred_proba),
        "log_loss": log_loss(y_test, y_pred_proba),
    }


def register_best_model(run_results):
    best = max(run_results, key=lambda item: item["metrics"]["roc_auc"])
    registered = mlflow.register_model(
        model_uri=best["model_uri"],
        name=REGISTERED_MODEL_NAME,
    )
    client = MlflowClient()
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="production",
        version=registered.version,
    )
    return best, registered.version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-register",
        action="store_true",
        help="Log runs only and skip MLflow Model Registry registration.",
    )
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models(y_train)
    run_results = []

    for model_name, spec in models.items():
        with mlflow.start_run(run_name=model_name):
            model = spec["model"]
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("data_path", str(DATA_PATH))
            mlflow.log_param("target_col", TARGET_COL)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_params(spec["params"])
            mlflow.log_metrics(metrics)

            model_info = mlflow.sklearn.log_model(model, name="model")
            run_results.append(
                {
                    "model_name": model_name,
                    "metrics": metrics,
                    "model_uri": model_info.model_uri,
                }
            )
            print(
                f"{model_name}: roc_auc={metrics['roc_auc']:.4f}, "
                f"f1={metrics['f1']:.4f}, uri={model_info.model_uri}"
            )

    if args.skip_register:
        return

    best, version = register_best_model(run_results)
    print(
        f"Best model: {best['model_name']} "
        f"roc_auc={best['metrics']['roc_auc']:.4f} "
        f"registered as {REGISTERED_MODEL_NAME} version {version} "
        "with alias production"
    )


if __name__ == "__main__":
    main()
