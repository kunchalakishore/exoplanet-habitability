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
    SELECT habitability_score
    FROM Exoplanet
    WHERE habitability_score IS NOT NULL
    """,
    conn
)

conn.close()

print(f"Total planets used: {len(df)}")

# -------------------------------
# Plot distribution
# -------------------------------
plt.figure(figsize=(8, 5))

sns.histplot(
    df["habitability_score"],
    bins=30,
    kde=True,
    color="#00c6ff"
)

plt.axvline(0.5, color="red", linestyle="--", label="Habitability Threshold")

plt.title("Habitability Score Distribution")
plt.xlabel("Habitability Score")
plt.ylabel("Number of Planets")
plt.legend()

plt.tight_layout()


# Save plot

plt.savefig("static/habitability_distribution.png", dpi=150)
plt.close()

print("✅ Habitability distribution plot saved:")
print("   static/habitability_distribution.png")
