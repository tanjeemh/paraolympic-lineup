import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
TEAM_OF_INTEREST = "Canada"
STINT_PATH = "data/raw/stint_data.csv"

# -----------------------------
# Load data
# -----------------------------
stints = pd.read_csv(STINT_PATH)

print("Columns found in CSV:")
print(list(stints.columns))

# -----------------------------
# Derive Canada-specific metrics
# -----------------------------
def compute_stint_metrics(row):
    if row["h_team"] == TEAM_OF_INTEREST:
        goals_for = row["h_goals"]
        goals_against = row["a_goals"]
        players = [row[f"home{i}"] for i in range(1, 5)]
    elif row["a_team"] == TEAM_OF_INTEREST:
        goals_for = row["a_goals"]
        goals_against = row["h_goals"]
        players = [row[f"away{i}"] for i in range(1, 5)]
    else:
        return None

    return {
        "players_on_court": ", ".join(players),
        "minutes": row["minutes"],
        "goals_for": goals_for,
        "goals_against": goals_against,
        "gd_per_min": (goals_for - goals_against) / row["minutes"]
    }

# Apply and FORCE DataFrame creation
derived = stints.apply(compute_stint_metrics, axis=1)
derived = derived.dropna()
derived = pd.DataFrame(list(derived))   # 🔑 THIS IS THE FIX

# -----------------------------
# Select rows for the figure
# -----------------------------
table_df = derived.head(10)

# -----------------------------
# Render table as a figure
# -----------------------------
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")

tbl = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    loc="center"
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)

plt.title(
    "Figure 1: Example of Stint-Level Data Structure\n"
    "Sample of Canada stints with derived goal differential per minute",
    pad=12
)

plt.tight_layout()
plt.savefig("figures/figure1_stint_level_data.png", dpi=300)
plt.show()
