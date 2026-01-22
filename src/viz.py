import pandas as pd
import plotly.express as px

def fig_player_impacts(player_impacts: pd.Series):
    df = player_impacts.reset_index()
    df.columns = ["player", "impact"]
    df = df.sort_values("impact", ascending=True).tail(20)  # show top 20
    fig = px.bar(df, x="impact", y="player", orientation="h", title="Top Player Impacts (Adjusted GD/min)")
    return fig

def fig_rating_vs_gd(team_df: pd.DataFrame):
    fig = px.scatter(
        team_df,
        x="total_rating",
        y="gd_per_min",
        color="opp_team",
        title="Stints: Total Rating vs GD/min (colored by opponent)",
        hover_data=["minutes", "gf", "ga", "is_home"]
    )
    return fig

def fig_minutes_distribution(minutes_played: dict):
    df = pd.DataFrame({"player": list(minutes_played.keys()), "minutes": list(minutes_played.values())})
    df = df.sort_values("minutes", ascending=False)
    fig = px.bar(df, x="player", y="minutes", title="Planned Minutes Distribution (Rotation Plan)")
    return fig

def fig_schedule_timeline(schedule_df: pd.DataFrame):
    # Build a Gantt-like plot: each block is a bar
    df = schedule_df.copy()
    df["task"] = "Lineup"
    fig = px.timeline(
        df,
        x_start="start_min",
        x_end="end_min",
        y="task",
        color="lineup",
        hover_data=["rating", "score"]
    )
    fig.update_layout(title="Rotation Plan Timeline (each block selects a lineup)", showlegend=False)
    fig.update_yaxes(visible=False)
    return fig
