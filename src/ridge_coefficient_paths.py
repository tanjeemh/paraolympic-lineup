import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


def compute_ridge_paths(
    X,
    y,
    w,
    alphas=None,
    max_players=20,
):
    """
    Compute ridge coefficient paths for player indicators.

    Parameters
    ----------
    X : pd.DataFrame
        Design matrix
    y : pd.Series
        Target (gd_per_min)
    w : pd.Series
        Sample weights (minutes)
    alphas : array-like
        Ridge regularization values
    max_players : int
        Number of player coefficients to plot (largest magnitude)

    Returns
    -------
    alphas, coef_df
    """

    if alphas is None:
        alphas = np.logspace(-2, 3, 30)

    coefs = []

    for a in alphas:
        model = Ridge(alpha=a)
        model.fit(X, y, sample_weight=w)
        coefs.append(model.coef_)

    coef_df = pd.DataFrame(
        coefs,
        columns=X.columns,
        index=alphas,
    )

    # keep only player indicators (drop controls + opponent effects)
    player_cols = [
        c for c in coef_df.columns
        if not c.startswith("opp_")
        and c not in ["total_rating", "is_home"]
    ]

    coef_df = coef_df[player_cols]

    # select top players by absolute magnitude at smallest alpha
    top_players = (
        coef_df.iloc[0]
        .abs()
        .sort_values(ascending=False)
        .head(max_players)
        .index
    )

    return alphas, coef_df[top_players]


def plot_ridge_paths(alphas, coef_df):
    """
    Plot ridge coefficient paths.
    """
    plt.figure(figsize=(10, 6))

    for col in coef_df.columns:
        plt.plot(alphas, coef_df[col], linewidth=1)

    plt.xscale("log")
    plt.xlabel("Ridge alpha (log scale)")
    plt.ylabel("Coefficient value")
    plt.title("Ridge Coefficient Paths (Player Effects)")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
