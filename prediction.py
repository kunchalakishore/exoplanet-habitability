import joblib
import numpy as np
from app import app, db, Exoplanet

# Load model + scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = [
    "pl_rade",
    "pl_bmasse",
    "pl_eqt",
    "pl_orbper",
    "st_teff",
    "st_rad"
]

with app.app_context():
    planets = Exoplanet.query.all()

    updated = 0
    skipped = 0

    for p in planets:
        values = [getattr(p, f) for f in FEATURES]

        if any(v is None for v in values):
            skipped += 1
            continue

        X = np.array([values])
        X_scaled = scaler.transform(X)

        score = float(model.predict(X_scaled)[0])
        p.habitability_score = score
        updated += 1

    db.session.commit()

    print("✅ Bulk prediction completed")
    print("Updated :", updated)
    print("Skipped :", skipped)
