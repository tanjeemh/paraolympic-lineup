import itertools
import numpy as np
import pandas as pd

def fatigue_factor(minutes_played: float, alpha: float, floor: float = 0.70):
    """
    Simple fatigue: impact decays as minutes increase.
    factor = max(floor, exp(-alpha * minutes_played))
    """
    return max(float(floor), float(np.exp(-alpha * float(minutes_played))))


def lineup_score_from_model(
    model,
    X_columns,
    lineup,
    rating_map,
    is_home=0,
    opp_dummy_cols=None,
    fatigue_alpha=0.02,
    minutes_played=None,
    equity_target=None,
    equity_lambda=0.0,
    overuse_lambda=0.0,
    max_minutes=None,
):
    """
    Build a single-row feature vector and predict GD/min.
    Then apply fatigue + equity penalties (coach constraints layer).
    """
    if minutes_played is None:
        minutes_played = {p: 0.0 for p in lineup}
    if equity_target is None:
        equity_target = {}
    if max_minutes is None:
        max_minutes = {}

    # Baseline prediction row
    row = pd.Series(0.0, index=X_columns, dtype=float)

    total_rating = 0.0
    for p in lineup:
        row[p] = 1.0
        total_rating += float(rating_map[p])

    row["total_rating"] = total_rating
    row["is_home"] = float(is_home)

    # Leave opponent dummies at 0 (reference opponent), unless caller sets one
    if opp_dummy_cols:
        for c in opp_dummy_cols:
            row[c] = 0.0

    base = float(model.predict(row.values.reshape(1, -1))[0])

    # Fatigue-adjusted impact: scale each player's contribution
    # We approximate by scaling base score by average fatigue of lineup.
    f_list = [fatigue_factor(minutes_played.get(p, 0.0), fatigue_alpha) for p in lineup]
    fatigue_scale = float(np.mean(f_list))
    fatigue_adjusted = base * fatigue_scale

    # Equity penalty: encourage minutes near target
    # penalty increases if player is below target (push them in) or above target (pull them out)
    eq_pen = 0.0
    for p in lineup:
        t = equity_target.get(p, None)
        if t is not None:
            diff = float(minutes_played.get(p, 0.0)) - float(t)
            eq_pen += abs(diff)

    eq_pen *= float(equity_lambda)

    # Overuse penalty: penalize exceeding max minutes
    ov_pen = 0.0
    for p in lineup:
        mx = max_minutes.get(p, None)
        if mx is not None:
            over = float(minutes_played.get(p, 0.0)) - float(mx)
            if over > 0:
                ov_pen += over

    ov_pen *= float(overuse_lambda)

    return fatigue_adjusted - eq_pen - ov_pen


def enumerate_valid_lineups(roster, rating_map, max_points=8.0):
    valid = []
    for combo in itertools.combinations(roster, 4):
        total = sum(float(rating_map[p]) for p in combo)
        if total <= float(max_points):
            valid.append((combo, total))
    return valid


def pick_best_lineup(
    model,
    X_columns,
    valid_lineups,
    rating_map,
    is_home=0,
    fatigue_alpha=0.02,
    minutes_played=None,
    equity_target=None,
    equity_lambda=0.0,
    overuse_lambda=0.0,
    max_minutes=None,
    opp_dummy_cols=None
):
    best = None
    best_score = -1e18

    for lineup, total in valid_lineups:
        score = lineup_score_from_model(
            model=model,
            X_columns=X_columns,
            lineup=lineup,
            rating_map=rating_map,
            is_home=is_home,
            opp_dummy_cols=opp_dummy_cols,
            fatigue_alpha=fatigue_alpha,
            minutes_played=minutes_played,
            equity_target=equity_target,
            equity_lambda=equity_lambda,
            overuse_lambda=overuse_lambda,
            max_minutes=max_minutes,
        )
        if score > best_score:
            best_score = score
            best = (lineup, total, score)

    return best  # (lineup tuple, total_rating, adjusted_score)


def simulate_rotation_plan(
    model,
    X_columns,
    roster,
    rating_map,
    game_minutes=32,
    block_minutes=1.0,
    max_points=8.0,
    injured=None,
    is_home=0,
    fatigue_alpha=0.02,
    min_minutes_per_player=0.0,
    max_minutes_per_player=32.0,
    equity_lambda=0.0,
    overuse_lambda=1.0,
):
    """
    Simple "coach planner" simulation:
    - time is divided into blocks
    - each block chooses the best lineup given injuries + fatigue + equity goals
    - ensures everyone approaches target playing time

    Returns:
      schedule_df: per block lineup selection
      minutes_played: dict of minutes per player
    """
    injured = set(injured or [])

    available = [p for p in roster if p not in injured]

    # targets: try to give everyone at least min_minutes, and cap at max
    # If min_minutes_per_player > 0, set a target to encourage playing time.
    # A simple target: average minutes, but at least the minimum.
    target = {}
    if len(available) > 0:
        avg_target = float(game_minutes) * 4.0 / float(len(available))  # total on-court minutes distributed
        for p in available:
            target[p] = max(float(min_minutes_per_player), avg_target)

    max_minutes = {p: float(max_minutes_per_player) for p in available}

    minutes_played = {p: 0.0 for p in available}

    valid_lineups = enumerate_valid_lineups(available, rating_map, max_points=max_points)

    rows = []
    t = 0.0
    n_blocks = int(np.ceil(float(game_minutes) / float(block_minutes)))

    for k in range(n_blocks):
        lineup, total, score = pick_best_lineup(
            model=model,
            X_columns=X_columns,
            valid_lineups=valid_lineups,
            rating_map=rating_map,
            is_home=is_home,
            fatigue_alpha=fatigue_alpha,
            minutes_played=minutes_played,
            equity_target=target,
            equity_lambda=equity_lambda,
            overuse_lambda=overuse_lambda,
            max_minutes=max_minutes,
        )

        # Update minutes
        for p in lineup:
            minutes_played[p] += float(block_minutes)

        rows.append({
            "block": k + 1,
            "start_min": t,
            "end_min": min(float(game_minutes), t + float(block_minutes)),
            "lineup": ", ".join(lineup),
            "rating": total,
            "score": score,
        })

        t += float(block_minutes)

    schedule_df = pd.DataFrame(rows)
    return schedule_df, minutes_played
