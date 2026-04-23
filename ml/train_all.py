"""
train_all.py
------------
Single entry-point that trains ALL models:
  1. Heart-rate regressor  (RandomForest → validated with CV)
  2. Anomaly detector      (IsolationForest on healthy subset)
  3. 5 × Disease classifiers (GradientBoosting with SMOTE + CV)

All models are wrapped in sklearn Pipelines (scaler → model) so the
backend never needs to remember to scale inputs — the pipeline handles it.

Run from the project root:
    python ml/train_all.py
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import (
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression  # noqa: F401 (kept for comparison)
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "processed" / "health_dataset.csv"
MODELS_DIR = ROOT / "ml" / "saved_models"
METRICS_PATH = MODELS_DIR / "training_metrics.json"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Features ──────────────────────────────────────────────────────────────────
BASE_FEATURES = ["steps", "temperature", "spo2", "glucose", "bp"]
DISEASE_FEATURES = ["steps", "temperature", "spo2", "glucose", "bp",
                    "heart_rate", "bmi", "age"]
DISEASE_TARGETS = {
    "diabetes":        "disease_diabetes_model.pkl",
    "hypertension":    "disease_hypertension_model.pkl",
    "hypoxia":         "disease_hypoxia_model.pkl",
    "fever_infection": "disease_fever_model.pkl",
    "cardiac_risk":    "disease_cardiac_model.pkl",
}

SEED = 42
metrics_log: dict = {}


def _banner(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── 1. Heart-rate regressor ───────────────────────────────────────────────────
def train_heart_rate(df: pd.DataFrame) -> None:
    _banner("Heart-rate regressor  (RandomForest + CV)")

    X = df[BASE_FEATURES + ["age", "bmi", "sex"]]
    y = df["heart_rate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=SEED,
        )),
    ])

    # 5-fold cross-validation on training set
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
    )
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)

    print(f"  CV MAE (5-fold): {-cv_scores.mean():.2f} ± {cv_scores.std():.2f} bpm")
    print(f"  Test  MAE:       {mae:.2f} bpm")
    print(f"  Test  R²:        {r2:.4f}")

    joblib.dump(pipeline, MODELS_DIR / "heart_model.pkl")
    metrics_log["heart_rate"] = {"cv_mae": round(-cv_scores.mean(), 3),
                                  "test_mae": round(mae, 3), "r2": round(r2, 4)}
    print("  ✅  heart_model.pkl saved")


# ── 2. Anomaly detector ───────────────────────────────────────────────────────
def train_anomaly(df: pd.DataFrame) -> None:
    _banner("Anomaly detector  (IsolationForest on healthy subset)")

    # Train only on records with no disease flag — cleaner decision boundary
    healthy_mask = (
        (df["diabetes"]       == 0) &
        (df["hypertension"]   == 0) &
        (df["hypoxia"]        == 0) &
        (df["fever_infection"]== 0) &
        (df["cardiac_risk"]   == 0)
    )
    X_healthy = df.loc[healthy_mask, BASE_FEATURES + ["age", "bmi"]]
    print(f"  Training on {len(X_healthy):,} healthy records "
          f"({len(X_healthy)/len(df)*100:.1f}% of dataset)")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            contamination=0.05,   # ~5% expected anomaly rate
            n_estimators=200,
            random_state=SEED,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X_healthy)

    joblib.dump(pipeline, MODELS_DIR / "anomaly_model.pkl")
    metrics_log["anomaly"] = {"trained_on_healthy_n": int(len(X_healthy)),
                               "contamination": 0.05}
    print("  ✅  anomaly_model.pkl saved")


# ── 3. Disease classifiers ────────────────────────────────────────────────────
def train_disease(df: pd.DataFrame, target: str, filename: str) -> None:
    X = df[DISEASE_FEATURES]
    y = df[target]

    pos_rate = y.mean()
    print(f"\n  [{target}]  pos={pos_rate*100:.1f}%", end="  ")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Manual over-sampling for minority class (replaces imblearn dependency)
    if pos_rate < 0.35:
        minority_mask = y_train == 1
        X_min = X_train[minority_mask]
        y_min = y_train[minority_mask]
        # Oversample to ~40% positive rate
        target_n = int(len(X_train) * 0.4 / (1 - 0.4))
        oversample_n = max(0, target_n - minority_mask.sum())
        if oversample_n > 0:
            idx = np.random.default_rng(SEED).integers(0, len(X_min), oversample_n)
            X_train = pd.concat([X_train, X_min.iloc[idx]], ignore_index=True)
            y_train = pd.concat([y_train, y_min.iloc[idx]], ignore_index=True)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            random_state=SEED,
        )),
    ])

    # Stratified 5-fold AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    auc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    test_auc = roc_auc_score(y_test, y_prob)

    print(f"  CV AUC={auc_scores.mean():.3f}±{auc_scores.std():.3f}  "
          f"Test AUC={test_auc:.3f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Healthy", "At-Risk"],
                                digits=3))

    joblib.dump(pipeline, MODELS_DIR / filename)
    metrics_log[target] = {
        "cv_auc_mean": round(auc_scores.mean(), 4),
        "cv_auc_std":  round(auc_scores.std(), 4),
        "test_auc":    round(test_auc, 4),
        "pos_rate":    round(pos_rate, 4),
    }
    print(f"  ✅  {filename} saved")


def train_all_diseases(df: pd.DataFrame) -> None:
    _banner("Disease classifiers  (GradientBoosting · 5-fold AUC · oversampling)")
    for target, filename in DISEASE_TARGETS.items():
        train_disease(df, target, filename)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Generate data if not present
    if not DATA_PATH.exists():
        print("Dataset not found — generating now …")
        sys.path.insert(0, str(Path(__file__).parent))
        from generate_data import generate
        df = generate()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"Dataset saved ({len(df):,} rows)")
    else:
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")

    train_heart_rate(df)
    train_anomaly(df)
    train_all_diseases(df)

    # Persist metrics for dashboard display
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_log, f, indent=2)

    _banner("All models trained successfully")
    print(f"  Metrics saved → {METRICS_PATH}\n")
