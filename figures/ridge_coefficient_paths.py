import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import Ridge
import os

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "data/raw/stint_data.csv"
TEAM = "Canada"
OUT_PATH = "figures/figure3_ridge_paths.png"

os.makedirs("figures", exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
stints = pd.read_csv(DATA_PATH)

# Filter Canada stints
stints = stints[
    (stints["h_team"] == TEAM) | (stints["a_team"] == TEAM)
].copy()

# -----------------------------
# Compute goals for / against
# -----------------------------
def compute_goals(row):
    if row["h_team"] == TEAM:
        return row["h_goals"], row["a_goals"]
    else:
        return row["a_goals"], row["h_goals"]

stints[["goals_for", "goals_against"]] = stints.apply(
    lambda r: pd.Series(compute_goals(r)),
    axis=1
)

stints["gd_per_min"] = (
    stints["goals_for"] - stints["goals_against"]
) / stints["minutes"]

# -----------------------------
# Extract players
# -----------------------------
def get_players(row):
    if row["h_team"] == TEAM:
        return [row[f"home{i}"] for i in range(1, 5)]
    else:
        return [row[f"away{i}"] for i in range(1, 5)]

stints["player_list"] = stints.apply(get_players, axis=1)

# -----------------------------
# Design matrix
# -----------------------------
mlb = MultiLabelBinarizer()
X = pd.DataFrame(
    mlb.fit_transform(stints["player_list"]),
    columns=mlb.classes_
)

y = stints["gd_per_min"].values

# -----------------------------
# Ridge paths
# -----------------------------
alphas = np.logspace(-2, 3, 30)
coefs = []

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    coefs.append(model.coef_)

coefs = np.array(coefs)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 6))

for i in range(min(8, coefs.shape[1])):
    plt.plot(alphas, coefs[:, i], label=X.columns[i])

plt.xscale("log")
plt.xlabel("Ridge Regularization Strength (α)")
plt.ylabel("Estimated Player Impact Coefficient")
plt.title("Figure 3: Effect of Ridge Regularization on Player Coefficients")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
plt.show()
