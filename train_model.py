"""
Train multiple ML models on the Pima Indians Diabetes dataset, evaluate them,
and save the best performer along with metrics/feature importance for the app.

Run:  python train_model.py
"""
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "diabetes.csv", header=None, names=COLUMNS)
    # Replace biologically impossible zeros with median (standard preprocessing)
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    return df


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.01, max_iter=250, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=0.1, random_state=42),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=42)),
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            use_label_encoder=False, eval_metric="logloss", random_state=42),
    }


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True),
    }


def main():
    print("Loading dataset...")
    df = load_data()
    print(f"   Shape: {df.shape} | Positives: {df.Outcome.sum()} / {len(df)}")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}
    trained = {}
    print("\nTraining models...")
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        trained[name] = model
        print(f"   {name:<20s} acc={metrics['accuracy']:.4f}  "
              f"f1={metrics['f1']:.4f}  auc={metrics['roc_auc']:.4f}")

    # Pick the most accurate model; use ROC-AUC as the tie-breaker because it
    # matters for imbalanced medical-risk data.
    best_name = max(results, key=lambda k: (results[k]["accuracy"], results[k]["roc_auc"]))
    best_model = trained[best_name]
    print(f"\nBest model: {best_name}")

    # Feature importance (Random Forest used for interpretability,
    # since some pipelines don't expose it directly)
    rf = trained["Random Forest"]
    importance = dict(zip(X.columns, rf.feature_importances_.tolist()))

    # ROC curve points for the best model
    proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_points = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    # Save artifacts
    with open(ROOT / "trained_model.sav", "wb") as f:
        pickle.dump(best_model, f)

    with open(ROOT / "model_metadata.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "feature_columns": X.columns.tolist(),
            "metrics_all": results,
            "metrics_best": results[best_name],
            "feature_importance": importance,
            "roc_curve": roc_points,
            "dataset_stats": {
                "rows": int(len(df)),
                "positives": int(y.sum()),
                "negatives": int((y == 0).sum()),
                "feature_means": X.mean().to_dict(),
                "feature_stds": X.std().to_dict(),
            },
        }, f, indent=2)

    print("Saved trained_model.sav and model_metadata.json")


if __name__ == "__main__":
    main()
