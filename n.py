# import pandas as pd
# import joblib
# from sklearn.preprocessing import StandardScaler
# from xgboost import XGBRegressor

# # -------------------------------------------------
# # Load dataset
# # -------------------------------------------------
# df = pd.read_csv("exoplanets_clean_full.csv")

# FEATURES = [
#     'pl_rade',
#     'pl_bmasse',
#     'pl_eqt',
#     'pl_orbper',
#     'st_teff',
#     'st_rad'
# ]

# TARGET = "habitability_score"  # or your computed label

# df = df.dropna(subset=FEATURES + [TARGET])

# X = df[FEATURES]
# y = df[TARGET]

# # -------------------------------------------------
# # Scaling
# # -------------------------------------------------
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # -------------------------------------------------
# # Train model
# # -------------------------------------------------
# model = XGBRegressor(
#     n_estimators=300,
#     max_depth=6,
#     learning_rate=0.05,
#     random_state=42
# )

# model.fit(X_scaled, y)

# # -------------------------------------------------
# # Save artifacts (IMPORTANT)
# # -------------------------------------------------
# joblib.dump(model, "model.pkl")
# joblib.dump(scaler, "scaler.pkl")
# joblib.dump(FEATURES, "features.pkl")

# print("✅ Model retrained with 6 features")
