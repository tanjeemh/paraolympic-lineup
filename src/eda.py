import pandas as pd
import plotly.express as px

def basic_eda(team_df):
    summary = {
        "num_stints": len(team_df),
        "avg_minutes": team_df["minutes"].mean(),
        "median_minutes": team_df["minutes"].median(),
        "mean_gd_per_min": team_df["gd_per_min"].mean(),
        "home_advantage": (
            team_df[team_df["is_home"] == 1]["gd_per_min"].mean()
            - team_df[team_df["is_home"] == 0]["gd_per_min"].mean()
        )
    }
    return summary


def fig_gd_distribution(team_df):
    return px.histogram(
        team_df,
        x="gd_per_min",
        nbins=30,
        title="Distribution of Goal Differential per Minute"
    )


def fig_player_participation(team_df):
    exploded = team_df.explode("players_on")
    counts = exploded["players_on"].value_counts().reset_index()
    counts.columns = ["player", "stints"]
    return px.bar(
        counts,
        x="player",
        y="stints",
        title="Player Participation (Number of Stints)"
    )
