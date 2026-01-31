import streamlit as st
import pandas as pd

from src.data_prep import (
    load_csvs,
    get_team_list,
    extract_team_view,
    build_rating_map,
    add_lineup_ratings,
    filter_min_stint_minutes,
)

from src.modeling import (
    build_design_matrix,
    fit_ridge,
    get_player_impacts,
    validate_ridge,
)

from src.optimization import (
    enumerate_valid_lineups,
    pick_best_lineup,
    simulate_rotation_plan,
)

from src.viz import (
    fig_player_impacts,
    fig_rating_vs_gd,
    fig_minutes_distribution,
    fig_schedule_timeline,
)

from src.milp_bridge import export_to_matlab, load_milp_solution


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Wheelchair Rugby Lineup Optimizer",
    layout="wide",
)

st.title("🏉 Wheelchair Rugby – Coach-Facing Lineup Optimizer")
st.caption(
    "Estimates player impact from stint-level data and optimizes "
    "Canada lineups under official classification rules."
)

# --------------------------------------------------
# Sidebar: Data
# --------------------------------------------------
st.sidebar.header("1) Data")

use_uploaded = st.sidebar.toggle("Upload CSVs instead of /data/raw", value=False)

if use_uploaded:
    stint_file = st.sidebar.file_uploader("Upload stint_data.csv", type=["csv"])
    player_file = st.sidebar.file_uploader("Upload player_data.csv", type=["csv"])

    if not (stint_file and player_file):
        st.stop()

    stints = pd.read_csv(stint_file)
    players_df = pd.read_csv(player_file)
else:
    stints, players_df = load_csvs(
        "data/raw/stint_data.csv",
        "data/raw/player_data.csv",
    )

teams = get_team_list(stints)
team_name = st.sidebar.selectbox(
    "Team to optimize",
    teams,
    index=teams.index("Canada") if "Canada" in teams else 0,
)

# --------------------------------------------------
# Sidebar: Model settings
# --------------------------------------------------
st.sidebar.header("2) Model settings")

min_stint_minutes = st.sidebar.slider(
    "Min stint minutes",
    0.0,
    5.0,
    0.5,
    0.1,
)

ridge_alpha = st.sidebar.slider(
    "Ridge alpha (stability)",
    0.0,
    50.0,
    1.0,
    0.5,
)

# --------------------------------------------------
# Build dataset
# --------------------------------------------------
team_df = extract_team_view(stints, team_name=team_name)
rating_map = build_rating_map(players_df)

team_df = add_lineup_ratings(team_df, rating_map)
team_df = filter_min_stint_minutes(team_df, min_stint_minutes)

# --------------------------------------------------
# Train model
# --------------------------------------------------
X, y, w, mlb = build_design_matrix(team_df)
model = fit_ridge(X, y, w, alpha=ridge_alpha)
player_impacts = get_player_impacts(model, X.columns, mlb.classes_)

# --------------------------------------------------
# Sidebar: Coach controls
# --------------------------------------------------
st.sidebar.header("3) Coach Scenario Controls")

roster = list(mlb.classes_)

injured = st.sidebar.multiselect("Injured / unavailable players", roster)

female_players = st.sidebar.multiselect(
    "Female players on roster (+0.5 rule)",
    roster,
)

max_points = st.sidebar.slider(
    "Base max classification points",
    6.0,
    8.0,
    8.0,
    0.5,
)

# Rotation controls (RESTORED)
st.sidebar.subheader("Rotation constraints")

fatigue_alpha = st.sidebar.slider(
    "Fatigue strength",
    0.0,
    0.10,
    0.02,
    0.005,
)

min_minutes_player = st.sidebar.slider(
    "Min minutes per player",
    0.0,
    16.0,
    0.0,
    0.5,
)

max_minutes_player = st.sidebar.slider(
    "Max minutes per player",
    4.0,
    32.0,
    32.0,
    0.5,
)

equity_lambda = st.sidebar.slider(
    "Fairness weight",
    0.0,
    2.0,
    0.5,
    0.1,
)

# --------------------------------------------------
# Export for MATLAB MILP
# --------------------------------------------------
export_to_matlab(
    player_impacts=player_impacts,
    rating_map=rating_map,
)


st.sidebar.success("Exported MILP inputs to matlab/")

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 Team Insights", "🏆 Best Lineups", "🔁 Rotation Planner"]
)

# ==================================================
# TAB 1 – Team insights
# ==================================================
with tab1:
    st.subheader("Player Impact Estimates (GD/min)")
    st.plotly_chart(fig_player_impacts(player_impacts), use_container_width=True)
    st.dataframe(player_impacts.reset_index(), use_container_width=True)

    st.subheader("Model validation")
    st.dataframe(validate_ridge(X, y, w), use_container_width=True)

    st.plotly_chart(fig_rating_vs_gd(team_df), use_container_width=True)

# ==================================================
# TAB 2 – Best Lineups (RESTORED)
# ==================================================
with tab2:
    available = [p for p in roster if p not in injured]

    valid = enumerate_valid_lineups(
        available,
        rating_map,
        max_points=max_points,
        female_players=female_players,
        female_bonus=0.5,
    )

    rows = []
    for lineup, total in valid:
        score = pick_best_lineup(
            model,
            X.columns,
            [(lineup, total)],
            rating_map,
            is_home=0.5,
            fatigue_alpha=0.0,
            minutes_played={p: 0.0 for p in available},
            equity_target=None,
            equity_lambda=0.0,
            max_minutes=None,
        )[2]

        rows.append(
            {
                "lineup": ", ".join(lineup),
                "rating": total,
                "pred_score": score,
            }
        )

    top_df = (
        pd.DataFrame(rows)
        .sort_values("pred_score", ascending=False)
        .head(20)
    )

    best = top_df.iloc[0]
    st.success(
        f"Top recommended lineup (fresh): "
        f"{best['lineup']} | rating={best['rating']} | score={best['pred_score']:.4f}"
    )

    st.markdown("### Top 20 predicted lineups (fresh)")
    st.dataframe(top_df, use_container_width=True)

    if st.button("Load MILP optimal lineup (MATLAB)"):
        milp = load_milp_solution()
        if milp:
            st.info(f"MILP optimal lineup: {milp}")
        else:
            st.warning("Run MATLAB solver first.")

# ==================================================
# TAB 3 – Rotation Planner (FULLY RESTORED)
# ==================================================
with tab3:
    st.caption(
        "Rotation planner uses heuristic optimization "
        "(MILP is used only for static lineup selection)."
    )

    game_minutes = st.number_input(
        "Game length (minutes)",
        min_value=8,
        max_value=64,
        value=32,
        step=1,
    )

    block_minutes = st.selectbox(
        "Decision block (minutes)",
        [0.5, 1.0, 2.0, 4.0],
        index=1,
    )

    if st.button("Generate rotation plan"):
        schedule_df, minutes_played = simulate_rotation_plan(
            model=model,
            X_columns=X.columns,
            roster=available,
            rating_map=rating_map,
            game_minutes=float(game_minutes),
            block_minutes=float(block_minutes),
            max_points=float(max_points),
            injured=injured,
            is_home=0.5,
            fatigue_alpha=fatigue_alpha,
            min_minutes_per_player=min_minutes_player,
            max_minutes_per_player=max_minutes_player,
            equity_lambda=equity_lambda,
        )

        st.subheader("Rotation schedule")
        st.dataframe(schedule_df, use_container_width=True)

        st.subheader("Minutes allocation")
        st.plotly_chart(
            fig_minutes_distribution(minutes_played),
            use_container_width=True,
        )

        st.plotly_chart(
            fig_schedule_timeline(schedule_df),
            use_container_width=True,
        )
