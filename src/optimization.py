"""
This module extends the static MILP solution by incorporating
practical coaching constraints:
- fatigue
- fairness
- substitution dynamics

This is NOT the core optimization model, but a decision-support layer.
"""

import itertools
import numpy as np
import pandas as pd
from collections import defaultdict


# =====================================================
# Fatigue model
# =====================================================
def fatigue_factor(minutes_played: float, alpha: float, floor: float = 0.70):
    """
    Simple fatigue: impact decays as minutes increase.
    factor = max(floor, exp(-alpha * minutes_played))
    """
    return max(float(floor), float(np.exp(-alpha * float(minutes_played))))


# =====================================================
# Lineup scoring (SOFT preferences only)
# =====================================================
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
    max_minutes=None,
    lineup_counts=None,
    repeat_lambda=0.0,
):
    """
    Predict GD/min for a lineup and apply SOFT preferences:
    - fatigue
    - fairness bonus (pulls in underused players)
    - lineup repetition penalty
    """

    minutes_played = minutes_played or {}
    equity_target = equity_target or {}
    lineup_counts = lineup_counts or {}

    # ---------------------------
    # Build prediction row
    # ---------------------------
    row = pd.Series(0.0, index=X_columns, dtype=float)

    total_rating = 0.0
    for p in lineup:
        row[p] = 1.0
        total_rating += float(rating_map[p])

    row["total_rating"] = total_rating
    row["is_home"] = float(is_home)

    if opp_dummy_cols:
        for c in opp_dummy_cols:
            row[c] = 0.0

    row_df = row.to_frame().T
    base_score = float(model.predict(row_df)[0])


    # ---------------------------
    # Fatigue adjustment
    # ---------------------------
    fatigue_factors = [
        fatigue_factor(minutes_played.get(p, 0.0), fatigue_alpha)
        for p in lineup
    ]
    fatigue_scale = float(np.mean(fatigue_factors))
    score = base_score * fatigue_scale

    # ---------------------------
    # Fairness BONUS (key fix)
    # ---------------------------
    fairness_bonus = 0.0
    for p in lineup:
        target = equity_target.get(p)
        if target is not None and minutes_played.get(p, 0.0) < target:
            fairness_bonus += (target - minutes_played[p])

    score += equity_lambda * fairness_bonus

    # ---------------------------
    # Lineup repetition penalty
    # ---------------------------
    lineup_key = tuple(sorted(lineup))
    repeats = lineup_counts.get(lineup_key, 0)
    score -= repeat_lambda * repeats

    return score


# =====================================================
# Enumerate valid 4-player lineups (8-point rule)
# =====================================================
def enumerate_valid_lineups(
    roster,
    rating_map,
    max_points=8.0,
    female_players=None,
    female_bonus=0.5,
):
    """
    Enumerate all valid 4-player lineups under classification rules.
    Applies +0.5 classification bonus if any female player is on court.
    """

    if female_players is None:
        female_players = set()
    else:
        female_players = set(female_players)

    valid = []

    from itertools import combinations

    for lineup in combinations(roster, 4):
        base_rating = sum(rating_map[p] for p in lineup)

        has_female = any(p in female_players for p in lineup)
        effective_rating = base_rating + (female_bonus if has_female else 0.0)

        if effective_rating <= max_points:
            valid.append((tuple(lineup), effective_rating))

    return valid



# =====================================================
# Pick best lineup (HARD + SOFT constraints)
# =====================================================
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
    max_minutes=None,
    opp_dummy_cols=None,
    lineup_counts=None,
    repeat_lambda=0.0,
):
    """
    Select the best lineup under:
    HARD constraints:
      - max minutes per player
    SOFT preferences:
      - fatigue
      - fairness bonus
      - repetition penalty
    """

    minutes_played = minutes_played or {}
    max_minutes = max_minutes or {}
    lineup_counts = lineup_counts or {}

    best = None
    best_score = -1e18

    for lineup, total in valid_lineups:

        # ---------------------------------
        # HARD CONSTRAINT: max minutes
        # ---------------------------------
        violates_cap = False
        for p in lineup:
            if minutes_played.get(p, 0.0) >= max_minutes.get(p, np.inf):
                violates_cap = True
                break

        if violates_cap:
            continue

        # ---------------------------------
        # Score lineup
        # ---------------------------------
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
            max_minutes=max_minutes,
            lineup_counts=lineup_counts,
            repeat_lambda=repeat_lambda,
        )

        if score > best_score:
            best_score = score
            best = (lineup, total, score)

    return best


# =====================================================
# Rotation / substitution simulation (FULL FIX)
# =====================================================
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
    repeat_lambda=8.0,
):
    """
    Coach-style rotation planner with:
    - HARD minute caps
    - Fairness-driven rotation
    - Lineup repetition control
    """

    injured = set(injured or [])
    available = [p for p in roster if p not in injured]

    # ---------------------------
    # Fairness targets
    # ---------------------------
    target = {}
    if available:
        avg_target = (float(game_minutes) * 4.0) / float(len(available))
        for p in available:
            target[p] = max(float(min_minutes_per_player), avg_target)

    # HARD max-minute caps
    max_minutes = {p: float(max_minutes_per_player) for p in available}

    minutes_played = {p: 0.0 for p in available}
    lineup_counts = defaultdict(int)

    valid_lineups = enumerate_valid_lineups(
        available, rating_map, max_points=max_points
    )

    rows = []
    t = 0.0
    n_blocks = int(np.ceil(float(game_minutes) / float(block_minutes)))

    for k in range(n_blocks):

        best = pick_best_lineup(
            model=model,
            X_columns=X_columns,
            valid_lineups=valid_lineups,
            rating_map=rating_map,
            is_home=is_home,
            fatigue_alpha=fatigue_alpha,
            minutes_played=minutes_played,
            equity_target=target,
            equity_lambda=equity_lambda,
            max_minutes=max_minutes,
            lineup_counts=lineup_counts,
            repeat_lambda=repeat_lambda,
        )

        if best is None:
            break  # no feasible lineup remains

        lineup, total, score = best

        # Update minutes
        for p in lineup:
            minutes_played[p] += float(block_minutes)

        lineup_key = tuple(sorted(lineup))
        lineup_counts[lineup_key] += 1

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
