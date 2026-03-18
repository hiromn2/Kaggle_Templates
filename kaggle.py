#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 15:06:06 2026

@author: hiro
"""

"""
Kaggle House Prices: Full, end-to-end implementation (tabular regression)

What this script does:
- Loads train.csv / test.csv
- Builds robust preprocessing (numeric + categorical) with sklearn Pipelines
- Uses log-target via TransformedTargetRegressor (recommended for House Prices)
- Trains + cross-validates:
    * Ridge baseline (strong)
    * Optional gradient boosting baseline (HistGradientBoostingRegressor)
    * Optional LightGBM / XGBoost / CatBoost if installed (auto-detected)
- Fits the best model by CV (lowest RMSE on log-target space)
- Predicts test.csv and writes submission.csv

Usage:
    python house_prices_full.py --train train.csv --test test.csv --target SalePrice --id_col Id --out submission.csv

Notes:
- Put train.csv and test.csv in the same folder, or pass paths explicitly.
- This code is designed to be safe against leakage and train/test mismatch.
"""

# What I understand
import numpy as np
import pandas as pd

import os
import sys #System-level hooks; in that script it’s imported but not essential (you can remove it unless you later want to handle exit codes, stdout/stderr behaviors, etc.).
import time #Timestamping logs and timing CV runs.
from sklearn.pipeline import Pipeline #Chains steps into one object with .fit() / .predict(). #Ensures preprocessing is applied consistently in training and testing.
from sklearn.preprocessing import OneHotEncoder, RobustScaler #Converts categorical columns to numeric indicator columns. # handle_unknown="ignore" prevents crashing when test contains categories not seen in training. #Scales numeric features using median and IQR (robust to outliers). # Helps linear models (Ridge/ElasticNet) because regularization depends on feature scale. # In the script it uses with_centering=False to keep things sparse-friendly.

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor #Applies different preprocessing pipelines to different column subsets.
from sklearn.impute import SimpleImputer #Fills missing values. Median or most frequent. add_indicator = True

# What I do not
from __future__ import annotations #Makes type annotations behave more cleanly # (treats annotations as strings internally), which avoids some forward-reference # issues and can reduce runtime overhead.
import argparse #Parses command-line arguments (--train, --test, --out, etc.).
import json #Writes results (CV scores, best model, params) to a .json log file.
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple



from pathlib import Path
from pprint import pprint
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import warnings
import json
import time


from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
    PolynomialFeatures,
    FunctionTransformer
)
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    GroupKFold,
    RepeatedKFold,
    cross_val_score,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier
)

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier

from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    f_classif,
    mutual_info_regression,
    mutual_info_classif,
    RFE
)
from sklearn.decomposition import PCA

from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.calibration import CalibratedClassifierCV





# -----------------------------
# Utilities
# -----------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_makedirs(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def detect_cols(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Detect numeric and categorical columns from a DataFrame.
    - Numeric: int/float/bool
    - Categorical: object/category
    """
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Preprocessor:
    - Numeric: median impute + missing indicator + robust scaling
    - Categorical: most_frequent impute + one-hot (ignore unknown)
    """
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", RobustScaler(with_centering=False)),  # sparse-friendly; with_centering=False
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",  # keep strict; avoid unexpected raw cols
        sparse_threshold=0.3,
    )
    return preprocess


def wrap_log_target(regressor) -> TransformedTargetRegressor:
    """
    House Prices best practice: train on log1p(SalePrice) and invert with expm1.
    This wrapper handles transform/inverse internally.
    """
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


@dataclass
class CVResult:
    name: str
    fold_scores: List[float]
    mean_score: float
    std_score: float


def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: KFold,
) -> CVResult:
    """
    Cross-validate an sklearn Pipeline.
    Score is RMSE on the original y-scale (since TransformedTargetRegressor inverts).
    """
    scores: List[float] = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = Pipeline(pipeline.steps)  # shallow clone-like; ok for our use
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        score = rmse(y_va.to_numpy(), np.asarray(preds))
        scores.append(score)
        print(f"  Fold {fold:02d}: RMSE = {score:.5f}")

    mean_s = float(np.mean(scores))
    std_s = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    return CVResult(
        name=str(pipeline.named_steps.get("model", "model")),
        fold_scores=scores,
        mean_score=mean_s,
        std_score=std_s,
    )


def maybe_import_boosting() -> Dict[str, object]:
    """
    Optionally detect external boosters if installed.
    Returns dict name -> estimator instance (not wrapped).
    """
    models: Dict[str, object] = {}

    # LightGBM
    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.01,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=0,
            n_jobs=-1,
        )
    except Exception:
        pass

    # XGBoost
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=5000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=0,
            n_jobs=-1,
        )
    except Exception:
        pass

    # CatBoost
    try:
        from catboost import CatBoostRegressor
        # We'll still one-hot encode, so CatBoost is not used in its native-cat mode here.
        # (Native-cat mode is a different pipeline design.)
        models["CatBoost"] = CatBoostRegressor(
            iterations=8000,
            learning_rate=0.02,
            depth=8,
            loss_function="RMSE",
            random_seed=0,
            verbose=False,
        )
    except Exception:
        pass

    return models


# -----------------------------
# Main training routine
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="train.csv")
    parser.add_argument("--test", type=str, default="test.csv")
    parser.add_argument("--target", type=str, default="SalePrice")
    parser.add_argument("--id_col", type=str, default="Id")
    parser.add_argument("--out", type=str, default="submission.csv")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--no_external_boosters", action="store_true")
    args = parser.parse_args()

    safe_makedirs(args.log_dir)

    # Load data
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    if args.target not in train_df.columns:
        raise ValueError(f"Target column '{args.target}' not found in train.")

    # Basic separation
    y = train_df[args.target].copy()
    X = train_df.drop(columns=[args.target])

    # Avoid using ID as feature
    if args.id_col in X.columns:
        X = X.drop(columns=[args.id_col])
    if args.id_col in test_df.columns:
        test_features = test_df.drop(columns=[args.id_col]).copy()
    else:
        test_features = test_df.copy()

    # Detect columns and build preprocess
    num_cols, cat_cols = detect_cols(X)
    preprocess = build_preprocessor(num_cols, cat_cols)

    # Define model candidates (all will be wrapped with log-target)
    candidates: Dict[str, Pipeline] = {}

    # Ridge baseline
    ridge = Ridge(alpha=10.0, random_state=0)
    ridge_t = wrap_log_target(ridge)
    candidates["Ridge(alpha=10)"] = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", ridge_t),
    ])

    # ElasticNet (sometimes beats Ridge)
    enet = ElasticNet(alpha=0.001, l1_ratio=0.1, random_state=0, max_iter=20000)
    enet_t = wrap_log_target(enet)
    candidates["ElasticNet(a=0.001,l1=0.1)"] = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", enet_t),
    ])

    # Sklearn gradient boosting baseline (strong, no external deps)
    hgb = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=None,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=0,
    )
    hgb_t = wrap_log_target(hgb)
    candidates["HistGBR(lr=0.05,leaves=31)"] = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", hgb_t),
    ])

    # Optional external boosters
    if not args.no_external_boosters:
        ext = maybe_import_boosting()
        for name, est in ext.items():
            candidates[name] = Pipeline(steps=[
                ("preprocess", preprocess),
                ("model", wrap_log_target(est)),
            ])

    # CV
    cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    print("\n=== Cross-validation (RMSE on original SalePrice scale) ===")
    all_results: Dict[str, CVResult] = {}
    for name, pipe in candidates.items():
        print(f"\nModel: {name}")
        start = time.time()
        res = cross_validate_pipeline(pipe, X, y, cv)
        elapsed = time.time() - start
        all_results[name] = CVResult(
            name=name,
            fold_scores=res.fold_scores,
            mean_score=res.mean_score,
            std_score=res.std_score,
        )
        print(f"  -> Mean RMSE = {res.mean_score:.5f} | Std = {res.std_score:.5f} | Time = {elapsed:.1f}s")

    # Pick best by mean RMSE
    best_name = min(all_results.keys(), key=lambda k: all_results[k].mean_score)
    best_res = all_results[best_name]
    print("\n=== Best model by CV ===")
    print(f"{best_name}: mean RMSE {best_res.mean_score:.5f} (std {best_res.std_score:.5f})")

    # Fit best on full data and predict test
    best_pipe = candidates[best_name]
    print("\nFitting best model on full training data...")
    best_pipe.fit(X, y)

    print("Predicting test...")
    test_preds = best_pipe.predict(test_features)

    # Create submission
    if args.id_col in test_df.columns:
        sub = pd.DataFrame({args.id_col: test_df[args.id_col], args.target: test_preds})
    else:
        # Kaggle expects Id; but if missing, still write predictions
        sub = pd.DataFrame({args.target: test_preds})

    sub.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")

    # Log results
    log_payload = {
        "train_path": args.train,
        "test_path": args.test,
        "target": args.target,
        "id_col": args.id_col,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "results": {
            k: {
                "fold_rmse": v.fold_scores,
                "mean_rmse": v.mean_score,
                "std_rmse": v.std_score,
            }
            for k, v in all_results.items()
        },
        "best": {
            "name": best_name,
            "mean_rmse": best_res.mean_score,
            "std_rmse": best_res.std_score,
        },
    }
    log_file = os.path.join(args.log_dir, f"cv_results_{int(time.time())}.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=2)
    print(f"Wrote CV log: {log_file}")


if __name__ == "__main__":
    main()
