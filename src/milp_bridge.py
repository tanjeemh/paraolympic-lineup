import os
import pandas as pd

MATLAB_DIR = "matlab"


def export_to_matlab(
    player_impacts: pd.Series,
    rating_map: dict,
):
    """
    Exports required inputs for MATLAB knapsack optimization
    (lecture formulation only).

    Outputs:
    - matlab/player_impacts.csv  (player, impact)
    - matlab/player_ratings.csv  (player, rating)
    """

    os.makedirs(MATLAB_DIR, exist_ok=True)

    # -----------------------------
    # Player impacts
    # -----------------------------
    impacts_df = (
        player_impacts
        .reset_index()
        .rename(columns={"index": "player", 0: "impact"})
    )

    impacts_df.to_csv(
        f"{MATLAB_DIR}/player_impacts.csv",
        index=False,
    )

    # -----------------------------
    # Player ratings
    # -----------------------------
    ratings_df = pd.DataFrame(
        {
            "player": list(player_impacts.index),
            "rating": [rating_map[p] for p in player_impacts.index],
        }
    )

    ratings_df.to_csv(
        f"{MATLAB_DIR}/player_ratings.csv",
        index=False,
    )


def load_milp_solution():
    """
    Reads optimal_lineup.csv produced by MATLAB
    """
    path = "matlab/optimal_lineup.csv"
    if not os.path.exists(path):
        return None

    sol = pd.read_csv(path)
    return sol.loc[sol["selected"] > 0.5, "player"].tolist()
