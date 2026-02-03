import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------------
# Shared evaluation helper
# --------------------------------------------------
def evaluate_model(model, X_train, X_test, y_train, y_test, w_train=None):
    if w_train is not None:
        model.fit(X_train, y_train, sample_weight=w_train)
    else:
        model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return {
        "mse": mean_squared_error(y_test, preds),
        "r2": r2_score(y_test, preds),
    }


# --------------------------------------------------
# OLS (Linear Regression) experiment
# --------------------------------------------------
def run_ols_experiment(X, y, w):
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.25, random_state=42
    )

    ols = LinearRegression()
    metrics = evaluate_model(
        ols, X_train, X_test, y_train, y_test, w_train
    )

    coef_variance = np.var(ols.coef_)

    return {
        "model": "OLS",
        "mse": metrics["mse"],
        "r2": metrics["r2"],
        "coef_variance": coef_variance,
    }


# --------------------------------------------------
# PCA + Linear Regression experiment
# --------------------------------------------------
def run_pca_experiment(X, y, w, explained_variance=0.90):
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.25, random_state=42
    )

    pca_pipeline = Pipeline([
        ("pca", PCA(n_components=explained_variance)),
        ("lr", LinearRegression())
    ])

    pca_pipeline.fit(X_train, y_train, lr__sample_weight=w_train)

    preds = pca_pipeline.predict(X_test)

    return {
        "model": "PCA + OLS",
        "mse": mean_squared_error(y_test, preds),
        "r2": r2_score(y_test, preds),
        "n_components": pca_pipeline.named_steps["pca"].n_components_,
        "explained_variance": pca_pipeline.named_steps["pca"]
            .explained_variance_ratio_.sum(),
    }


# --------------------------------------------------
# Ridge baseline (for comparison)
# --------------------------------------------------
def run_ridge_experiment(X, y, w, alpha=1.0):
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.25, random_state=42
    )

    ridge = Ridge(alpha=alpha)
    metrics = evaluate_model(
        ridge, X_train, X_test, y_train, y_test, w_train
    )

    return {
        "model": f"Ridge (α={alpha})",
        "mse": metrics["mse"],
        "r2": metrics["r2"],
    }


# --------------------------------------------------
# Run all experiments
# --------------------------------------------------
def run_all_experiments(X, y, w, ridge_alpha=1.0):
    results = [
        run_ols_experiment(X, y, w),
        run_pca_experiment(X, y, w),
        run_ridge_experiment(X, y, w, alpha=ridge_alpha),
    ]

    return pd.DataFrame(results).sort_values("mse")
