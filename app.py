import streamlit as st
import pandas as pd

from src.data_prep import (
    load_csvs,
    get_team_list,
    extract_team_view,
    build_rating_map,
    add_lineup_ratings,
    filter_min_stint_minutes
)
from src.modeling import build_design_matrix, fit_ridge, get_player_impacts
from src.optimization import (
    enumerate_valid_lineups,
    pick_best_lineup,
    simulate_rotation_plan
)
from src.viz import (
    fig_player_impacts,
    fig_rating_vs_gd,
    fig_minutes_distribution,
    fig_schedule_timeline
)

st.set_page_config(page_title="Wheelchair Rugby Lineup Optimizer", layout="wide")

st.title("🏉 Wheelchair Rugby – Coach-Facing Lineup Optimizer")
st.caption("Uses stint-level goals + minutes + lineups + ratings to estimate player impacts and generate lineup/rotation recommendations under the 8-point rule.")

# ---------- Sidebar: Data ----------
st.sidebar.header("1) Data")

use_uploaded = st.sidebar.toggle("Upload CSVs instead of /data/raw", value=False)

if use_uploaded:
    stint_file = st.sidebar.file_uploader("Upload stint_data.csv", type=["csv"])
    player_file = st.sidebar.file_uploader("Upload player_data.csv", type=["csv"])
    if stint_file and player_file:
        stints = pd.read_csv(stint_file)
        players = pd.read_csv(player_file)
    else:
        st.warning("Upload both CSVs to proceed.")
        st.stop()
else:
    stints, players = load_csvs("data/raw/stint_data.csv", "data/raw/player_data.csv")

teams = get_team_list(stints)
if not teams:
    st.error("No teams found in stint data.")
    st.stop()

team_name = st.sidebar.selectbox("Team to optimize", teams, index=teams.index("Canada") if "Canada" in teams else 0)

# ---------- Sidebar: Modeling ----------
st.sidebar.header("2) Model settings")
min_stint_minutes = st.sidebar.slider("Min stint minutes to include", 0.0, 5.0, 0.5, 0.1)
ridge_alpha = st.sidebar.slider("Ridge alpha (stability)", 0.0, 50.0, 1.0, 0.5)

# ---------- Build team data ----------
team_df = extract_team_view(stints, team_name=team_name)
rating_map = build_rating_map(players)
team_df = add_lineup_ratings(team_df, rating_map)
team_df = filter_min_stint_minutes(team_df, min_stint_minutes)

if team_df.empty:
    st.error("No stints remain after filtering (team name / ratings missing / min minutes too high).")
    st.stop()

# ---------- Train model ----------
X, y, w, mlb = build_design_matrix(team_df)
model = fit_ridge(X, y, w, alpha=ridge_alpha)
player_impacts = get_player_impacts(model, X.columns, mlb.classes_)

# ---------- Sidebar: Scenario / Coach controls ----------
st.sidebar.header("3) Coach Scenario Controls")

roster = list(mlb.classes_)
injured = st.sidebar.multiselect("Injured / unavailable players", roster, default=[])

max_points = st.sidebar.slider("Max lineup points", 6.0, 8.0, 8.0, 0.5)

is_home = st.sidebar.selectbox("Assume home/away context for recommendations", ["Neutral", "Home", "Away"], index=0)
is_home_val = 0.5 if is_home == "Neutral" else (1.0 if is_home == "Home" else 0.0)

st.sidebar.subheader("Fatigue & fairness")
fatigue_alpha = st.sidebar.slider("Fatigue strength (higher = faster decay)", 0.0, 0.10, 0.02, 0.005)

min_minutes_player = st.sidebar.slider("Min minutes per player (goal)", 0.0, 16.0, 0.0, 0.5)
max_minutes_player = st.sidebar.slider("Max minutes per player (cap)", 4.0, 32.0, 32.0, 0.5)

equity_lambda = st.sidebar.slider("Fairness weight (encourage rotation)", 0.0, 2.0, 0.5, 0.1)
overuse_lambda = st.sidebar.slider("Overuse penalty weight", 0.0, 5.0, 1.0, 0.1)

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["📊 Team Insights", "🏆 Best Lineups", "🔁 Rotation Planner"])

# ===== Tab 1: Team insights =====
with tab1:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Player Impact Estimates (Adjusted GD/min)")
        st.plotly_chart(fig_player_impacts(player_impacts), use_container_width=True)
        st.dataframe(player_impacts.reset_index().rename(columns={"index":"player", 0:"impact"}), use_container_width=True)

    with c2:
        st.subheader("Stint Performance Landscape")
        st.plotly_chart(fig_rating_vs_gd(team_df), use_container_width=True)
        st.write("Tip: Look for clusters where GD/min is high even at lower total ratings → efficient lineups.")

    st.subheader("Raw stint sample (team-centric)")
    show_cols = ["game_id","minutes","gf","ga","gd","gd_per_min","is_home","opp_team","total_rating","players_on"]
    existing = [c for c in show_cols if c in team_df.columns]
    st.dataframe(team_df[existing].head(50), use_container_width=True)

# ===== Tab 2: Best lineups =====
with tab2:
    st.subheader("Best Lineups Under Constraints")
    available_roster = [p for p in roster if p not in set(injured)]
    if len(available_roster) < 4:
        st.error("Not enough available players to form a lineup of 4.")
        st.stop()

    valid = enumerate_valid_lineups(available_roster, rating_map, max_points=max_points)

    if not valid:
        st.error("No valid 4-player lineups under the selected max points.")
        st.stop()

    best = pick_best_lineup(
        model=model,
        X_columns=X.columns,
        valid_lineups=valid,
        rating_map=rating_map,
        is_home=is_home_val,
        fatigue_alpha=fatigue_alpha,
        minutes_played={p: 0.0 for p in available_roster},
        equity_target=None,
        equity_lambda=0.0,
        overuse_lambda=0.0,
        max_minutes=None
    )

    st.success(f"Top recommended lineup (fresh, no rotation constraints): {best[0]} | rating={best[1]:.1f} | score={best[2]:.4f}")

    # Show top N lineups for review
    st.markdown("### Top 20 predicted lineups (fresh)")
    rows = []
    for lineup, total in valid:
        # Score without fairness penalties here so coaches see pure performance
        score = pick_best_lineup(
            model, X.columns, [(lineup, total)], rating_map,
            is_home=is_home_val, fatigue_alpha=fatigue_alpha,
            minutes_played={p: 0.0 for p in available_roster},
            equity_target=None, equity_lambda=0.0, overuse_lambda=0.0, max_minutes=None
        )[2]
        rows.append({"lineup": ", ".join(lineup), "rating": total, "pred_score": score})

    top_df = pd.DataFrame(rows).sort_values("pred_score", ascending=False).head(20)
    st.dataframe(top_df, use_container_width=True)

# ===== Tab 3: Rotation planner =====
with tab3:
    st.subheader("Rotation / Substitution Planner")
    st.caption("Simulates a full game by selecting a lineup each time block while applying fatigue + playing-time fairness constraints.")

    c1, c2, c3 = st.columns(3)
    with c1:
        game_minutes = st.number_input("Game length (minutes)", min_value=8, max_value=64, value=32, step=1)
    with c2:
        block_minutes = st.selectbox("Decision block (minutes)", [0.5, 1.0, 2.0, 4.0], index=1)
    with c3:
        run = st.button("Generate rotation plan")

    if run:
        schedule_df, minutes_played = simulate_rotation_plan(
            model=model,
            X_columns=X.columns,
            roster=available_roster,
            rating_map=rating_map,
            game_minutes=float(game_minutes),
            block_minutes=float(block_minutes),
            max_points=float(max_points),
            injured=injured,
            is_home=is_home_val,
            fatigue_alpha=float(fatigue_alpha),
            min_minutes_per_player=float(min_minutes_player),
            max_minutes_per_player=float(max_minutes_player),
            equity_lambda=float(equity_lambda),
            overuse_lambda=float(overuse_lambda),
        )

        st.markdown("### Rotation schedule")
        st.dataframe(schedule_df, use_container_width=True)

        st.markdown("### Minutes allocation")
        st.plotly_chart(fig_minutes_distribution(minutes_played), use_container_width=True)
        st.dataframe(
            pd.DataFrame({"player": list(minutes_played.keys()), "minutes": list(minutes_played.values())})
              .sort_values("minutes", ascending=False),
            use_container_width=True
        )

        st.markdown("### Timeline")
        st.plotly_chart(fig_schedule_timeline(schedule_df), use_container_width=True)

        st.info("Coach tip: Increase fairness weight to spread minutes, increase fatigue strength to force more substitutions.")
