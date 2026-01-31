import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data/raw/stint_data.csv"
TEAM = "Canada"
OUT_PATH = "figures/figure2_design_matrix.png"

# -----------------------------
# Load data
# -----------------------------
stints = pd.read_csv(DATA_PATH)

# -----------------------------
# Filter Canada stints
# -----------------------------
canada = stints[
    (stints["h_team"] == TEAM) | (stints["a_team"] == TEAM)
].copy()

# -----------------------------
# Extract Canada players
# -----------------------------
def get_players(row):
    if row["h_team"] == TEAM:
        return [row[f"home{i}"] for i in range(1, 5)]
    else:
        return [row[f"away{i}"] for i in range(1, 5)]

canada["player_list"] = canada.apply(get_players, axis=1)

# -----------------------------
# Build design matrix
# -----------------------------
mlb = MultiLabelBinarizer()
X = pd.DataFrame(
    mlb.fit_transform(canada["player_list"]),
    columns=mlb.classes_
)

# Small subset for visualization
X_sample = X.iloc[:15, :10]

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(12, 6))
sns.heatmap(
    X_sample,
    cmap="Blues",
    cbar=False,
    linewidths=0.5
)

plt.xlabel("Players")
plt.ylabel("Stints")
plt.title("Figure 2: Design Matrix Representation (Binary Player Indicators)")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
plt.show()
