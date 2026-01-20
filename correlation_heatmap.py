import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# Ensure static folder exists
# -------------------------------
os.makedirs("static", exist_ok=True)

# -------------------------------
# Load data from database
# -------------------------------
conn = sqlite3.connect("instance/exoplanets.db")

df = pd.read_sql_query(
    """
    SELECT
        pl_rade,
        pl_bmasse,
        pl_eqt,
        pl_orbper,
        st_teff,
        st_rad,
        habitability_score
    FROM Exoplanet
    WHERE habitability_score IS NOT NULL
    """,
    conn
)

conn.close()

print("Loaded rows:", len(df))

# -------------------------------
# Correlation matrix
# -------------------------------
corr = df.corr(numeric_only=True)

# -------------------------------
# Plot heatmap
# -------------------------------
plt.figure(figsize=(9, 7))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Correlation Strength"}
)

plt.title("Star–Planet Feature Correlation Heatmap")
plt.tight_layout()

# -------------------------------
# Save plot
# -------------------------------
plt.savefig("static/feature_correlation_heatmap.png", dpi=150)
plt.close()

print("✅ Correlation heatmap saved:")
print("   static/feature_correlation_heatmap.png")
