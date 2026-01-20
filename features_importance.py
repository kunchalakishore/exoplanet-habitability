import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# Ensure static folder exists
# -------------------------------
os.makedirs("static", exist_ok=True)

# -------------------------------
# Load model & features
# -------------------------------
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

# -------------------------------
# Create importance DataFrame
# -------------------------------
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance_df)

# -------------------------------
# Plot (Seaborn-compliant)
# -------------------------------
plt.figure(figsize=(8, 5))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature",
    hue="Feature",        # ✅ FIX
    palette="viridis",
    legend=False          # ✅ FIX
)

plt.title("Feature Importance – Habitability Model")
plt.xlabel("Importance Score")
plt.ylabel("Feature")

plt.tight_layout()

# -------------------------------
# Save plot
# -------------------------------
plt.savefig("static/feature_importance.png", dpi=150)
plt.close()

print("\n✅ Feature importance plot saved at:")
print("   static/feature_importance.png")
