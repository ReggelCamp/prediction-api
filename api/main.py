# from fastapi import FastAPI
# import joblib
# from fastapi.middleware.cors import CORSMiddleware
# import os
# from mangum import Mangum
# import numpy as np


# # 1. Create app
# app = FastAPI()

# # 2. Add middleware (allow frontend access)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # 3. Set model path (your folder)
# # MODEL_PATH = r"C:\Users\REGGEL\OneDrive\Desktop\mahine_learning"
# # print("✅ Using model path:", MODEL_PATH)

# # # 4. Load model & encoders
# # model = joblib.load(os.path.join(MODEL_PATH, "new_disease_case_predictor.pkl"))
# # month_encoder = joblib.load(os.path.join(MODEL_PATH, "new_month_encoder.pkl"))
# # disease_encoder = joblib.load(os.path.join(MODEL_PATH, "new_disease_encoder.pkl"))
# # season_encoder = joblib.load(os.path.join(MODEL_PATH, "season_encoder.pkl"))

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "PKL-files")

# model = joblib.load(os.path.join(MODEL_PATH, "new_disease_case_predictor.pkl"))
# month_encoder = joblib.load(os.path.join(MODEL_PATH, "new_month_encoder.pkl"))
# disease_encoder = joblib.load(os.path.join(MODEL_PATH, "new_disease_encoder.pkl"))
# season_encoder = joblib.load(os.path.join(MODEL_PATH, "season_encoder.pkl"))

# # 5. Define month → season mapping
# def get_season(month: str) -> str:
#     month = month.lower()
#     if month in ["december", "january", "february"]:
#         return "Wet"     # Northeast monsoon / dry season
#     elif month in ["march", "april", "may"]:
#         return "Dry"      # Hot dry season
#     elif month in ["june", "july", "august", "september", "october", "november"]:
#         return "Wet"        # Rainy season
#     else:
#         return "unknown"

# # 6. Prediction endpoint
# # @app.get("/predict")
# # def predict(disease_name: str, case_count: float, month: str, year: int):
# #     try:
# #         # Encode categorical inputs
# #         disease_encoded = disease_encoder.transform([disease_name])[0]
# #         month_encoded = month_encoder.transform([month])[0]

# #         # Determine and encode season
# #         season = get_season(month)
# #         season_encoded = season_encoder.transform([season])[0]

# #         # Simulate missing engineered features
# #         # You can later replace these with real previous or rolling values
# #         prev_month_cases = case_count  # temporary assumption
# #         rolling3 = case_count          # temporary assumption
# #         time_index = 1                 # default; could track per disease

# #         # Combine all inputs in the same order as model training
# #         input_data = np.array([[
# #             disease_encoded,
# #             month_encoded,
# #             year,
# #             season_encoded,
# #             prev_month_cases,
# #             rolling3,
# #             time_index
# #         ]])

# #         # Predict outcome
# #         prediction = model.predict(input_data)[0]

# #         return {
# #             "prediction": round(float(prediction), 2),
# #             "season": season
# #         }

# #     except Exception as e:
# #         return {"error": str(e)}

# @app.get("/predict")
# def predict(disease_name: str, case_count: float, month: str, year: int):
#     try:
#         disease_encoded = disease_encoder.transform([disease_name])[0]
#         month_encoded = month_encoder.transform([month])[0]
#         season = get_season(month)
#         season_encoded = season_encoder.transform([season])[0]

#         prev_month_cases = case_count
#         rolling3 = case_count
#         time_index = 1

#         input_data = np.array([[
#             disease_encoded,
#             month_encoded,
#             year,
#             season_encoded,
#             prev_month_cases,
#             rolling3,
#             time_index
#         ]])

#         predicted_cases = model.predict(input_data)[0]

#         # 🔥 Binary classification: 1 = rise, 0 = no rise
#         # is_rise = 1 if(predicted_cases + (predicted_cases * 0.30 ))> case_count else 0
#         is_rise = 1 if ((predicted_cases - case_count) / case_count) > 0.30 else 0

#         return {
#             "prediction": int(is_rise),
#             "predicted_cases": round(float(predicted_cases), 2),
#             "season": season
#         }

#     except Exception as e:
#         return {"error": str(e)}

# handler = Mangum(app)


#copy
# import os
# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List, Dict
# import numpy as np
# import joblib
# import traceback

# app = FastAPI()

# # ---------- LOAD MODEL + ENCODERS ----------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "PKL-files")

# model            = joblib.load(os.path.join(MODEL_PATH, "updated_disease_case_predictor.pkl"))
# month_encoder    = joblib.load(os.path.join(MODEL_PATH, "updated_month_encoder.pkl"))
# disease_encoder  = joblib.load(os.path.join(MODEL_PATH, "updated_disease_encoder.pkl"))
# season_encoder   = joblib.load(os.path.join(MODEL_PATH, "updated_season_encoder.pkl")) 
# # ---------- SEASON HELPER ----------
# def get_season(month: str) -> str:
#     month = month.lower()
#     if month in ["december", "january", "february"]:
#         return "Wet"
#     if month in ["march", "april", "may"]:
#         return "Dry"
#     if month in ["june", "july", "august", "september", "october", "november"]:
#         return "Wet"
#     return "unknown"

# # ---------- REQUEST SCHEMA ----------
# class PredictRequest(BaseModel):
#     diseases: List[str]
#     disease_case_counts: Dict[str, float]   # 👈 per-disease totals from UI
#     month: str
#     year: int

# # ---------- ENDPOINT ----------
# @app.post("/predict_multiple")
# def predict_multiple(req: PredictRequest):
#     try:
#         month_encoded = month_encoder.transform([req.month])[0]
#         season      = get_season(req.month)
#         season_encoded = season_encoder.transform([season])[0]

#         results = []

#         for disease_name in req.diseases:
#             case_count = req.disease_case_counts.get(disease_name, 0)

#             try:
#                 disease_encoded = disease_encoder.transform([disease_name])[0]
#             except ValueError:
#                 return {"error": f"Unknown disease: {disease_name}"}

#             # synthetic features (same logic as before)
#             prev_month_cases = case_count * 0.9
#             rolling3 = (case_count + prev_month_cases + case_count * 1.1) / 3
#             time_index = 1

#             input_row = np.array([[
#                 disease_encoded,
#                 month_encoded,
#                 req.year,
#                 season_encoded,
#                 prev_month_cases,
#                 rolling3,
#                 time_index
#             ]])

#             predicted_cases = float(model.predict(input_row)[0])
#             is_rise = 1 if predicted_cases > case_count * 1.25 else 0

#             results.append({
#                 "disease_name": disease_name,
#                 "predicted_cases": round(predicted_cases, 2),
#                 "prediction": is_rise
#             })

#         return {"results": results, "season": season}

#     except Exception as e:
#         print("❌ Internal error:", traceback.format_exc())
#         return {"error": f"Internal Server Error: {str(e)}"}

#copy 2
import os, json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
import joblib
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "PKL-files")

if not firebase_admin._apps:
    cred_json = os.getenv("FIREBASE_CREDENTIALS")
    if not cred_json:
        raise ValueError("FIREBASE_CREDENTIALS environment variable is not set")
    cred_dict = json.loads(cred_json)
    if isinstance(cred_dict.get("private_key"), str):
        cred_dict["private_key"] = cred_dict["private_key"].replace('\\n', '\n')
    cred = credentials.Certificate(cred_dict)
    initialize_app(cred)

db = firestore.client()

model = joblib.load(os.path.join(MODEL_PATH, "updated_disease_case_predictor.pkl"))
month_encoder = joblib.load(os.path.join(MODEL_PATH, "updated_month_encoder.pkl"))
disease_encoder = joblib.load(os.path.join(MODEL_PATH, "updated_disease_encoder.pkl"))
season_encoder = joblib.load(os.path.join(MODEL_PATH, "updated_season_encoder.pkl"))

class MultiPredictionRequest(BaseModel):
    diseases: List[str]
    disease_case_counts: Dict[str, float]
    month: str
    year: int
    municipality: str

SEASON_MAP = {
    'December': 'Wet', 'January': 'Wet', 'February': 'Dry', 'March': 'Dry',
    'April': 'Dry', 'May': 'Dry', 'June': 'Wet', 'July': 'Wet',
    'August': 'Wet', 'September': 'Wet', 'October': 'Wet', 'November': 'Wet'
}

def get_season(month: str) -> str:
    return SEASON_MAP.get(month, "Wet")

def safe_transform(encoder, values):
    known = set(encoder.classes_)
    transformed = []
    for v in values:
        if v in known:
            transformed.append(encoder.transform([v])[0])
        else:
            transformed.append(-1)  # unseen label
    return np.array(transformed)

def fetch_monthly_series(municipality: str, disease: str) -> pd.DataFrame:
    docs = (
        db.collection_group("UploadedCases")
            .where("Municipality", "==", municipality.strip().title())
            .where("DiseaseName", "==", disease.strip().title())
            .stream()
    )

    rows = []
    for d in docs:
        data = d.to_dict()
        if not data.get("DateReported") or data.get("CaseCount") is None:
            continue

        try:
            month_key = datetime.fromisoformat(data["DateReported"].replace("Z", "+00:00")).strftime("%Y-%m")
            cases = int(data["CaseCount"])
        except:
            continue

        rows.append({
            "month": month_key,
            "cases": cases
        })

    if not rows:
        return pd.DataFrame(columns=["month", "cases"])

    df = (
        pd.DataFrame(rows)
          .groupby("month", as_index=False)["cases"]
          .sum()
          .sort_values("month")
    )
    return df

def get_month_key(year, month_name):
    month_num = datetime.strptime(month_name, "%B").month
    return f"{year}-{month_num:02d}"

def get_previous_month_key(year, month_name):
    month_num = datetime.strptime(month_name, "%B").month
    cur = datetime(year, month_num, 1)
    prev = cur - relativedelta(months=1)
    return prev.strftime("%Y-%m")

def get_last_3_month_keys(year, month_name):
    month_num = datetime.strptime(month_name, "%B").month
    cur = datetime(year, month_num, 1)
    keys = []
    for i in range(1, 4):
        m = cur - relativedelta(months=i)
        keys.append(m.strftime("%Y-%m"))
    return keys

def build_features_from_frontend(diseases, disease_case_counts, month, year, municipality):
    features = {}

    month_encoded = safe_transform(month_encoder, [month])[0]
    season_encoded = safe_transform(season_encoder, [get_season(month)])[0]

    if month_encoded == -1:
        month_encoded = 0
    if season_encoded == -1:
        season_encoded = 0

    prev_key = get_previous_month_key(year, month)
    last3_keys = get_last_3_month_keys(year, month)

    for disease in diseases:
        disease_encoded = safe_transform(disease_encoder, [disease])[0]
        if disease_encoded == -1:
            continue

        # Get monthly series
        df = fetch_monthly_series(municipality, disease)

        if df.empty:
            prev_cases = 0.0
            rolling3 = 0.0
            time_index = 0
        else:
            # Previous month
            prev_row = df[df["month"] == prev_key]
            prev_cases = float(prev_row["cases"].iloc[0]) if not prev_row.empty else 0.0

            # Rolling 3-month
            r3_rows = df[df["month"].isin(last3_keys)]
            rolling3 = float(r3_rows["cases"].mean()) if not r3_rows.empty else 0.0

            # Time index
            sorted_months = sorted(df["month"].unique())
            target_key = get_month_key(year, month)
            time_index = sorted_months.index(target_key) if target_key in sorted_months else len(sorted_months)

        # Feature order
        features[disease] = [
            disease_encoded,
            month_encoded,
            year,
            season_encoded,
            prev_cases,     
            rolling3,       
            time_index      
        ]

    return features


@app.post("/predict_multiple")
def predict_multiple(request: MultiPredictionRequest):
    try:
        inputs = build_features_from_frontend(
            request.diseases,
            request.disease_case_counts,
            request.month,
            request.year,
            request.municipality
        )

        if not inputs:
            return {"error": "No valid diseases to predict", "results": []}

        results = []

        for disease, feature_row in inputs.items():
            predicted_cases = max(0, round(float(model.predict([feature_row])[0])))

            # Fetch for historical average for display
            df = fetch_monthly_series(request.municipality, disease)
            current_cases = float(df["cases"].mean()) if not df.empty else 0.0

            # Rising classification
            if current_cases > 0:
                increase = ((predicted_cases - current_cases) / current_cases) * 100
                prediction = 1 if increase >= 25 else 0
            else:
                prediction = 1 if predicted_cases > 0 else 0

            results.append({
                "disease_name": disease,
                "prediction": prediction,
                "probability": min(100, max(0, predicted_cases)),
                "predicted_cases": predicted_cases,
                "current_cases": current_cases
            })

        results.sort(key=lambda x: x["probability"], reverse=True)
        rising = [r for r in results if r["prediction"] == 1]

        return {
            "target_month": request.month,
            "target_year": request.year,
            "results": results,
            "top_disease": rising[0]["disease_name"] if rising else None,
            "rising_count": len(rising)
        }

    except Exception as e:
        print("❌ predict_multiple error:", e)
        return {"error": f"Internal server error: {str(e)}", "results": []}
