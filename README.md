# Wheelchair Rugby Coach Dashboard (Streamlit)

## Overview
This project implements a coach-facing decision support system for wheelchair rugby lineup optimization.  
The tool estimates individual player impact from stint-level game data, evaluates valid four-player lineups under the official 8-point classification rule, and supports both static lineup selection and dynamic rotation planning.

The application is designed to bridge **descriptive analytics**, **predictive modeling**, and **prescriptive optimization** in a single interactive workflow.

## What This Tool Does
- Learns player on-court impact from stint-level data (goals, minutes, lineups)
- Estimates player contributions using **ridge regression**
- Enumerates and ranks valid 4-player lineups under classification constraints
- Supports coach scenarios:
  - Injured / unavailable players
  - Fatigue-aware rotation planning
  - Fairness and minute allocation constraints
- Optionally integrates a **MATLAB MILP solver** for exact lineup optimization
- Provides interactive visualizations and tables via Streamlit

## Setup
1) Put CSV files in:
   data/raw/stint_data.csv
   data/raw/player_data.csv

2) Install:
   pip install -r requirements.txt

3) Run:
   streamlit run app.py
