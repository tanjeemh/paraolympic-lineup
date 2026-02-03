# 🏉 Wheelchair Rugby Lineup Optimization System

## Overview

This project implements a coach-facing decision support system for wheelchair rugby lineup optimization.

The system estimates individual player impact from stint-level game data, evaluates valid four-player lineups under the official 8-point classification rule, and supports both static lineup selection and dynamic rotation planning.

The application bridges:
- Descriptive analytics (stint-level summaries)
- Predictive modeling (adjusted plus-minus via ridge regression)
- Prescriptive optimization (heuristic rotation planning and exact MILP optimization)

---

## Repository Structure
```
PARALYMPIC-LINEUP/
│
├── app.py                     # Streamlit coach dashboard
├── prediction.ipynb           # Exploratory analysis & prediction plots
├── requirements.txt
├── README.md
│
├── data/
│   └── raw/
│       ├── stint_data.csv     # Stint-level game data
│       └── player_data.csv    # Player ratings / metadata
│
├── figures/                   # Scripts + generated figures
│   ├── stint_table.py
│   ├── design_matrix.py
│   ├── ridge_coefficient_paths.py
│   ├── figure1_stint_level_data.png
│   ├── figure2_design_matrix.png
│   └── figure3_ridge_paths.png
│
├── matlab/                    # MATLAB MILP optimization
│   ├── solve_canada_lineup_milp.m
│   ├── player_impacts.csv
│   ├── player_ratings.csv
│   ├── optimal_lineup.csv
│   └── milp_summary.csv
│
└── src/                       # Core Python modules
    ├── data_prep.py
    ├── modeling.py
    ├── optimization.py
    ├── milp_bridge.py
    ├── viz.py
    └── model_experiments.py
```
---

## Environment Setup

Install dependencies from the project root:

    pip install -r requirements.txt

---

## Running the Streamlit Dashboard

The Streamlit application is the primary interface for interacting with the model.

Run the app using:

    streamlit run app.py

The dashboard provides:
- Player impact estimates (adjusted goal differential per minute)
- Ridge regression validation results
- Ranked valid four-player lineups
- Coach scenario controls:
  - Injured / unavailable players
  - Classification point cap
  - Fatigue strength
  - Fairness and minute constraints
- Full game rotation planning
- Export of model outputs for MATLAB optimization

---

## Generating Analysis Figures (Figures 1&2 and Table 1)

The following scripts generate standalone, publication-ready figures used in analysis and reporting.

### Table — Stint-Level Data Structure

Run:

    python figures/stint_table.py

Output file:

    figures/figure1_stint_level_data.png

This figure shows how raw stint data is transformed into modeling-ready observations.

---

### Figure 1 — Design Matrix Representation

Run:

    python figures/design_matrix.py

Output file:

    figures/figure2_design_matrix.png

This figure visualizes the binary player-indicator design matrix used for ridge regression.

---

### Figure 2 — Ridge Coefficient Paths

Run:

    python figures/ridge_coefficient_paths.py

Output file:

    figures/figure3_ridge_paths.png

This figure demonstrates how ridge regularization stabilizes player impact estimates.

---

## Running the MATLAB MILP Optimization

The MATLAB solver computes the exact optimal four-player lineup under the classification constraint.

Steps:
1. Open MATLAB
2. Navigate to the matlab/ directory (or add it to the MATLAB path)
3. Set the matlab/ folder as the current working directory
4. Run the following command in the MATLAB Command Window:

    solve_canada_lineup_milp

The script:
- Loads player_impacts.csv and player_ratings.csv
- Constructs a Mixed-Integer Linear Program (MILP)
- Solves for the optimal four-player lineup
- Enforces the official classification points constraint
- Outputs:
  - optimal_lineup.csv
  - milp_summary.csv

The Streamlit application can load and display this MILP-optimal lineup for comparison.

---

## Running the Prediction Notebook

The notebook contains exploratory analysis and supporting plots.

Launch it using:

    jupyter notebook prediction.ipynb

The notebook includes:
- Distribution of goal differential per minute
- Player participation statistics
- Model comparison experiments
- Supporting validation plots

The notebook is not required to run the Streamlit dashboard.

---

## Modeling & Optimization Summary

- Model: Ridge regression (adjusted plus-minus)
- Target variable: Goal differential per minute
- Features:
  - Player on-court indicators
  - Total classification rating
  - Home/away indicator
  - Opponent fixed effects
- Optimization:
  - Python heuristic rotation planning
  - MATLAB MILP for exact static lineup optimization
- Visualization: Streamlit, Plotly, Matplotlib

---

## Intended Use

This system is designed as a decision support tool for evaluating lineup performance under realistic competitive constraints.
It prioritizes interpretability, transparency, and reproducibility over black-box automation.
