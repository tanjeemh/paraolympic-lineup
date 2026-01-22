import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import Ridge

def build_design_matrix(team_df: pd.DataFrame):
    """
    Features:
      - player indicators (1 if on court)
      - controls: total_rating, is_home
      - opponent dummies (fixed effects)
    Target:
      - gd_per_min
    """
    mlb = MultiLabelBinarizer()
    X_players = mlb.fit_transform(team_df["players_on"])
    X_players = pd.DataFrame(X_players, columns=mlb.classes_, index=team_df.index)

    X_controls = pd.DataFrame({
        "total_rating": team_df["total_rating"],
        "is_home": team_df["is_home"],
    }, index=team_df.index)

    X_opp = pd.get_dummies(team_df["opp_team"], prefix="opp", drop_first=True)

    X = pd.concat([X_players, X_controls, X_opp], axis=1)
    y = team_df["gd_per_min"]
    w = team_df["minutes"]

    return X, y, w, mlb


def fit_ridge(X, y, w, alpha: float = 1.0):
    model = Ridge(alpha=float(alpha))
    model.fit(X, y, sample_weight=w)
    return model


def get_player_impacts(model, X_columns, player_names):
    coefs = pd.Series(model.coef_, index=X_columns)
    impacts = coefs[player_names].sort_values(ascending=False)
    return impacts
