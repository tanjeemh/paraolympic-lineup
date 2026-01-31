import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def build_design_matrix(team_df: pd.DataFrame):
    """
    Build the design matrix for adjusted plus-minus modeling.

    Features:
      - Player indicators (1 if player is on the court)
      - Controls:
          * total_rating (classification sum)
          * is_home (home/away context)
      - Opponent fixed effects (dummy variables)

    Target:
      - gd_per_min (goal differential per minute)

    Weights:
      - minutes (longer stints carry more information)
    """

    # Player on-court indicators
    mlb = MultiLabelBinarizer()
    X_players = mlb.fit_transform(team_df["players_on"])
    X_players = pd.DataFrame(
        X_players, columns=mlb.classes_, index=team_df.index
    )

    # Control variables
    X_controls = pd.DataFrame(
        {
            "total_rating": team_df["total_rating"],
            "is_home": team_df["is_home"],
        },
        index=team_df.index,
    )

    # Opponent fixed effects
    X_opp = pd.get_dummies(
        team_df["opp_team"], prefix="opp", drop_first=True
    )

    # Full design matrix
    X = pd.concat([X_players, X_controls, X_opp], axis=1)

    # Target and weights
    y = team_df["gd_per_min"]
    w = team_df["minutes"]

    return X, y, w, mlb


def fit_ridge(X, y, w, alpha: float = 1.0):
    """
    Fit a ridge regression model for adjusted plus-minus.

    Ridge regression is used instead of OLS because player indicators
    are highly collinear (players always appear in fixed-size lineups).
    The L2 penalty stabilizes coefficient estimates.
    """
    model = Ridge(alpha=float(alpha))
    model.fit(X, y, sample_weight=w)
    return model


def validate_ridge(
    X,
    y,
    w,
    alphas=(0.0, 0.1, 1.0, 5.0, 10.0),
    test_size=0.25,
    random_state=42,
):
    """
    Validate ridge regression by comparing different alpha values.

    Returns a DataFrame with out-of-sample MSE for each alpha.
    """

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X,
        y,
        w,
        test_size=test_size,
        random_state=random_state,
    )

    results = []

    for a in alphas:
        model = Ridge(alpha=a)
        model.fit(X_train, y_train, sample_weight=w_train)

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        results.append(
            {
                "alpha": a,
                "mse": mse,
            }
        )

    return pd.DataFrame(results).sort_values("alpha")


def get_player_impacts(model, X_columns, player_names):
    """
    Extract player impact estimates (adjusted GD/min).

    Only coefficients corresponding to player indicators are returned.
    """
    coefs = pd.Series(model.coef_, index=X_columns)
    impacts = coefs[player_names].sort_values(ascending=False)
    return impacts
