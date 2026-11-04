"""
Model training script for the breast cancer triage service.

Reads the DVC-tracked CSV, trains multiple model families with cross-validation,
selects the best by malignant-class recall, evaluates on a held-out test set,
and serializes the winning pipeline + model card.
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sklearn


def load_params(path="params.yaml"):
    """Load parameters from params.yaml."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_pipelines(params):
    """Build candidate sklearn Pipelines from params."""
    lr_params = params["models"]["logistic_regression"]
    rf_params = params["models"]["random_forest"]

    pipelines = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=lr_params["C"],
                max_iter=lr_params["max_iter"],
                solver=lr_params["solver"],
                random_state=params["seed"],
                class_weight="balanced",
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=rf_params["n_estimators"],
                max_depth=rf_params["max_depth"],
                min_samples_split=rf_params["min_samples_split"],
                random_state=params["seed"],
                class_weight="balanced",
            )),
        ]),
    }
    return pipelines


def evaluate_cv(pipeline, X_train, y_train, params):
    """Run stratified cross-validation and return mean metrics.
    
    Note: In this dataset, target 0 = malignant (positive class),
    target 1 = benign. We measure malignant recall = recall for class 0.
    """
    cv = StratifiedKFold(
        n_splits=params["cv_folds"],
        shuffle=True,
        random_state=params["seed"],
    )

    # We use pos_label=0 because malignant is class 0
    scoring = {
        "recall_malignant": "recall",  # will be set with make_scorer below
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
    }

    # For recall of the malignant class (label 0), we need a custom scorer
    from sklearn.metrics import make_scorer

    def malignant_recall(y_true, y_pred):
        return recall_score(y_true, y_pred, pos_label=0)

    scoring = {
        "recall_malignant": make_scorer(malignant_recall),
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
    }

    results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

    return {
        "mean_recall_malignant": float(np.mean(results["test_recall_malignant"])),
        "mean_accuracy": float(np.mean(results["test_accuracy"])),
        "mean_roc_auc": float(np.mean(results["test_roc_auc"])),
    }


def evaluate_test(pipeline, X_test, y_test):
    """Evaluate on the held-out test set. Malignant = class 0."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    # For ROC AUC in binary classification, sklearn needs P(positive_label).
    # Since y_true uses {0, 1} and default positive_label=1, we pass P(class 1)
    # which is the benign probability. The AUC is symmetric so this is correct.
    benign_prob_col = list(pipeline.classes_).index(1)

    return {
        "test_recall_malignant": float(recall_score(y_test, y_pred, pos_label=0)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob[:, benign_prob_col])),
    }


def main():
    parser = argparse.ArgumentParser(description="Train breast cancer triage model")
    parser.add_argument("--data-path", default="data/raw/breast_cancer.csv",
                        help="Path to the DVC-tracked CSV dataset")
    parser.add_argument("--output-dir", default="artifacts",
                        help="Directory for model artifacts")
    parser.add_argument("--model-version", default="1.0.0",
                        help="Semantic version tag for the model")
    args = parser.parse_args()

    # Load parameters
    params = load_params()
    seed = params["seed"]

    # Read the dataset
    print(f"Reading dataset from {args.data_path}")
    df = pd.read_csv(args.data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    feature_names = list(X.columns)
    print(f"Dataset: {len(df)} samples, {len(feature_names)} features")

    # Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=seed, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Build and evaluate candidate pipelines
    pipelines = build_pipelines(params)
    cv_results = {}
    all_metrics = {}

    print("\n--- Cross-Validation Results ---")
    for name, pipeline in pipelines.items():
        metrics = evaluate_cv(pipeline, X_train, y_train, params)
        cv_results[name] = metrics
        print(f"{name}:")
        print(f"  Recall (malignant): {metrics['mean_recall_malignant']:.4f}")
        print(f"  Accuracy:           {metrics['mean_accuracy']:.4f}")
        print(f"  ROC AUC:            {metrics['mean_roc_auc']:.4f}")

    # Select best model by malignant recall
    best_name = max(cv_results, key=lambda k: cv_results[k]["mean_recall_malignant"])
    print(f"\nSelected model: {best_name} (highest malignant recall)")

    # Train the selected model on the full training set
    best_pipeline = pipelines[best_name]
    best_pipeline.fit(X_train, y_train)

    # Evaluate on test set
    test_metrics = evaluate_test(best_pipeline, X_test, y_test)
    print("\n--- Test Set Results ---")
    print(f"  Recall (malignant): {test_metrics['test_recall_malignant']:.4f}")
    print(f"  Accuracy:           {test_metrics['test_accuracy']:.4f}")
    print(f"  ROC AUC:            {test_metrics['test_roc_auc']:.4f}")

    # Gate: fail if malignant recall is below 0.95
    if test_metrics["test_recall_malignant"] < 0.95:
        print(f"\nFAIL: Test-set malignant recall {test_metrics['test_recall_malignant']:.4f} < 0.95")
        sys.exit(1)

    # Collect all metrics for the metrics file
    all_metrics = {
        "selected_model": best_name,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
    }

    # Save artifacts
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = os.path.join(args.output_dir, "model.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"\nModel saved to {model_path}")

    # Model card
    model_card = {
        "model_version": args.model_version,
        "model_family": best_name,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "cv_metrics": cv_results[best_name],
        "test_metrics": test_metrics,
        "decision_threshold": params["decision_threshold"],
    }
    card_path = os.path.join(args.output_dir, "model_card.json")
    with open(card_path, "w") as f:
        json.dump(model_card, f, indent=2)
    print(f"Model card saved to {card_path}")

    # Metrics file for DVC
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
