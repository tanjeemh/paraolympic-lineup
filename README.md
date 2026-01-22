# Wheelchair Rugby Coach Dashboard (Streamlit)

## What this does
- Learns player on-court impact from stint data (goals/minutes + lineups)
- Predicts best 4-player lineups under the 8-point rule
- Allows coach scenarios:
  - injuries/unavailable players
  - fatigue over time
  - min/max minutes and fairness rotation
- Generates a rotation plan + charts

## Setup
1) Put CSV files in:
   data/raw/stint_data.csv
   data/raw/player_data.csv

2) Install:
   pip install -r requirements.txt

3) Run:
   streamlit run app.py
