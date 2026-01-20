# # -------------------------------------------------
# # Load Dataset into Database (ONE-TIME SCRIPT)
# # -------------------------------------------------

# import pandas as pd
# from flask import Flask
# from flask_sqlalchemy import SQLAlchemy

# # -------------------------------------------------
# # Flask & Database Setup
# # -------------------------------------------------
# app = Flask(__name__)

# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///exoplanets.db"
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# db = SQLAlchemy(app)

# # -------------------------------------------------
# # Database Model (MUST MATCH app.py)
# # -------------------------------------------------
# class Exoplanet(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(100), unique=True, nullable=False)

#     pl_rade = db.Column(db.Float)
#     pl_bmasse = db.Column(db.Float)
#     pl_eqt = db.Column(db.Float)
#     pl_orbper = db.Column(db.Float)
#     st_teff = db.Column(db.Float)
#     st_rad = db.Column(db.Float)
#     st_lum = db.Column(db.Float)
#     sy_dist = db.Column(db.Float)

#     habitability = db.Column(db.Integer)
#     habitability_probability = db.Column(db.Float)

# # -------------------------------------------------
# # MAIN EXECUTION
# # -------------------------------------------------
# if __name__ == "__main__":

#     with app.app_context():

#         # Create tables
#         db.create_all()

#         # Load cleaned dataset
#         df = pd.read_csv("exoplanets_clean_full.csv")

#         print("Columns in CSV:")
#         print(df.columns.tolist())

#         inserted = 0
#         skipped = 0

#         for _, row in df.iterrows():

#             planet_name = row["pl_name"]   # ✅ CORRECT COLUMN

#             # Avoid duplicates
#             exists = Exoplanet.query.filter_by(name=planet_name).first()
#             if exists:
#                 skipped += 1
#                 continue

#             planet = Exoplanet(
#                 name=planet_name,          # ✅ FIXED
#                 pl_rade=row["pl_rade"],
#                 pl_bmasse=row["pl_bmasse"],
#                 pl_eqt=row["pl_eqt"],
#                 pl_orbper=row["pl_orbper"],
#                 st_teff=row["st_teff"],
#                 st_rad=row["st_rad"],
#                 st_lum=row["st_lum"],
#                 sy_dist=row["sy_dist"]
#             )

#             db.session.add(planet)
#             inserted += 1

#         db.session.commit()

#         print("✅ DATA LOADING COMPLETED")
#         print(f"Inserted planets : {inserted}")
#         print(f"Skipped duplicates: {skipped}")
# import joblib
# import pandas as pd
# from app import app, db, Exoplanet

# FEATURES = [
#     "pl_rade", "pl_bmasse", "pl_eqt", "pl_orbper",
#     "st_teff", "st_rad", "st_lum", "sy_dist"
# ]

# model = joblib.load("model.pkl")

# with app.app_context():

#     planets = Exoplanet.query.all()
#     updated = 0

#     for p in planets:
#         try:
#             # ✅ Use DataFrame with column names
#             X = pd.DataFrame([{
#                 "pl_rade": p.pl_rade,
#                 "pl_bmasse": p.pl_bmasse,
#                 "pl_eqt": p.pl_eqt,
#                 "pl_orbper": p.pl_orbper,
#                 "st_teff": p.st_teff,
#                 "st_rad": p.st_rad,
#                 "st_lum": p.st_lum,
#                 "sy_dist": p.sy_dist
#             }])

#             prob = model.predict_proba(X)[0][1]

#             p.habitability_probability = float(prob)
#             p.habitability = int(prob >= 0.5)
#             updated += 1

#         except Exception as e:
#             print(f"Skipping {p.name}: {e}")

#     db.session.commit()
#     print(f"✅ Predictions completed for {updated} planets")
# import pandas as pd
# from app import app, db, Exoplanet

# CSV_FILE = "exoplanets_clean_full.csv"

# FEATURES = [
#     "pl_rade",
#     "pl_bmasse",
#     "pl_eqt",
#     "pl_orbper",
#     "st_teff",
#     "st_rad"
# ]

# with app.app_context():

#     # 🔥 IMPORTANT: fresh database (RUN ONCE)
#     db.drop_all()
#     db.create_all()
#     print("✅ Fresh database created")

#     df = pd.read_csv(CSV_FILE)

#     inserted = 0
#     skipped = 0

#     for _, row in df.iterrows():

#         name = row["pl_name"]

#         # Skip duplicates
#         if Exoplanet.query.filter_by(name=name).first():
#             skipped += 1
#             continue

#         # Skip rows with missing required features
#         if row[FEATURES].isnull().any():
#             skipped += 1
#             continue

#         planet = Exoplanet(
#             name=name,
#             pl_rade=row["pl_rade"],
#             pl_bmasse=row["pl_bmasse"],
#             pl_eqt=row["pl_eqt"],
#             pl_orbper=row["pl_orbper"],
#             st_teff=row["st_teff"],
#             st_rad=row["st_rad"]
#         )

#         db.session.add(planet)
#         inserted += 1

#     db.session.commit()

#     print("✅ DATA LOADED SUCCESSFULLY")
#     print("Inserted:", inserted)
#     print("Skipped :", skipped)

import pandas as pd
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# -------------------------------------------------
# Setup
# -------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(
    BASE_DIR, "instance", "exoplanets.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# -------------------------------------------------
# DB Model (MUST MATCH app.py)
# -------------------------------------------------
class Exoplanet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)

    pl_rade = db.Column(db.Float)
    pl_bmasse = db.Column(db.Float)
    pl_eqt = db.Column(db.Float)
    pl_orbper = db.Column(db.Float)
    st_teff = db.Column(db.Float)
    st_rad = db.Column(db.Float)

    habitability_score = db.Column(db.Float)
    rank = db.Column(db.Integer)

# -------------------------------------------------
# LOAD CSV
# -------------------------------------------------
if __name__ == "__main__":

    df = pd.read_csv("exoplanets_clean_full.csv")  # your CSV

    print("CSV Columns:")
    print(df.columns.tolist())

    with app.app_context():

        db.create_all()

        inserted = 0
        skipped = 0

        for idx, row in df.iterrows():

            # 🔑 AUTO-GENERATED NAME
            planet_name = f"Planet_{idx+1}"

            # Skip if any required value missing
            values = [
                row["pl_rade"],
                row["pl_bmasse"],
                row["pl_eqt"],
                row["pl_orbper"],
                row["st_teff"],
                row["st_rad"],
            ]

            if any(pd.isna(v) for v in values):
                skipped += 1
                continue

            planet = Exoplanet(
                name=planet_name,
                pl_rade=row["pl_rade"],
                pl_bmasse=row["pl_bmasse"],
                pl_eqt=row["pl_eqt"],
                pl_orbper=row["pl_orbper"],
                st_teff=row["st_teff"],
                st_rad=row["st_rad"],
            )

            db.session.add(planet)
            inserted += 1

        db.session.commit()

        print("✅ CSV LOADING COMPLETED")
        print(f"Inserted planets : {inserted}")
        print(f"Skipped rows     : {skipped}")
