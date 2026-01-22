import pandas as pd
import numpy as np

HOME_PLAYERS = ["home1", "home2", "home3", "home4"]
AWAY_PLAYERS = ["away1", "away2", "away3", "away4"]

def load_csvs(stint_path: str, player_path: str):
    stints = pd.read_csv(stint_path)
    players = pd.read_csv(player_path)

    stints.columns = stints.columns.str.strip()
    players.columns = players.columns.str.strip()

    # numeric coercion
    for col in ["minutes", "h_goals", "a_goals"]:
        if col in stints.columns:
            stints[col] = pd.to_numeric(stints[col], errors="coerce")

    stints = stints.dropna(subset=["minutes", "h_goals", "a_goals"])
    stints = stints[stints["minutes"] > 0].copy()

    # player ratings numeric
    if "rating" in players.columns:
        players["rating"] = pd.to_numeric(players["rating"], errors="coerce")

    players = players.dropna(subset=["player", "rating"]).copy()

    return stints, players


def get_team_list(stints: pd.DataFrame):
    teams = set()
    if "h_team" in stints.columns: teams.update(stints["h_team"].dropna().unique().tolist())
    if "a_team" in stints.columns: teams.update(stints["a_team"].dropna().unique().tolist())
    return sorted(list(teams))


def extract_team_view(stints: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """
    Converts each raw stint into a team-centric row:
      - players_on: the 4 players for team_name
      - gf/ga: goals for/against in that stint
      - is_home: 1 if team is home
      - opp_team: opponent label
    """
    # team is home
    home_df = stints[stints["h_team"] == team_name].copy()
    home_df["is_home"] = 1
    home_df["opp_team"] = home_df["a_team"]
    home_df["gf"] = home_df["h_goals"]
    home_df["ga"] = home_df["a_goals"]
    home_df["players_on"] = home_df[HOME_PLAYERS].values.tolist()

    # team is away
    away_df = stints[stints["a_team"] == team_name].copy()
    away_df["is_home"] = 0
    away_df["opp_team"] = away_df["h_team"]
    away_df["gf"] = away_df["a_goals"]
    away_df["ga"] = away_df["h_goals"]
    away_df["players_on"] = away_df[AWAY_PLAYERS].values.tolist()

    df = pd.concat([home_df, away_df], ignore_index=True)

    df["gd"] = df["gf"] - df["ga"]
    df["gd_per_min"] = df["gd"] / df["minutes"]
    df["gf_per_min"] = df["gf"] / df["minutes"]
    df["ga_per_min"] = df["ga"] / df["minutes"]

    return df


def build_rating_map(players: pd.DataFrame) -> dict:
    return players.set_index("player")["rating"].to_dict()


def add_lineup_ratings(team_df: pd.DataFrame, rating_map: dict) -> pd.DataFrame:
    def lineup_rating(lst):
        vals = [rating_map.get(p, np.nan) for p in lst]
        if any(pd.isna(v) for v in vals):
            return np.nan
        return float(np.sum(vals))

    out = team_df.copy()
    out["total_rating"] = out["players_on"].apply(lineup_rating)
    out = out.dropna(subset=["total_rating"]).copy()
    return out


def filter_min_stint_minutes(team_df: pd.DataFrame, min_minutes: float) -> pd.DataFrame:
    return team_df[team_df["minutes"] >= float(min_minutes)].copy()
