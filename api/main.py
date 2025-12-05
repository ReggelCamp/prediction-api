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
import os,json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
import joblib
import numpy as np
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv  # Add this import

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Add CORS middleware BEFORE any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "PKL-files")

# Load .env only if it exists (for local development)
if os.path.exists(".env"):
    load_dotenv()

# Initialize Firebase
if not firebase_admin._apps:
    cred_json = os.getenv("FIREBASE_CREDENTIALS")
    
    if not cred_json:
        raise ValueError("FIREBASE_CREDENTIALS environment variable is not set")
    
    # Parse the JSON string
    cred_dict = json.loads(cred_json)
    
    # Fix the newlines in the private key if needed
    if isinstance(cred_dict.get("private_key"), str):
        cred_dict["private_key"] = cred_dict["private_key"].replace('\\n', '\n')
    
    cred = credentials.Certificate(cred_dict)
    firebase_app = initialize_app(cred)

db = firestore.client()

#  Load Model + Encoders

model = joblib.load(os.path.join(MODEL_PATH, "updated_disease_case_predictor.pkl"))
month_encoder = joblib.load(os.path.join(MODEL_PATH, "updated_month_encoder.pkl"))
disease_encoder = joblib.load(os.path.join(MODEL_PATH, "updated_disease_encoder.pkl"))
season_encoder = joblib.load(os.path.join(MODEL_PATH, "updated_season_encoder.pkl"))


#  PREDICTION REQUEST BODY

class MultiPredictionRequest(BaseModel):
    diseases: List[str]
    disease_case_counts: Dict[str, float]
    month: str
    year: int
    municipality: str


#  Helper: Compute Season

SEASON_MAP = {
    'December': 'Wet', 'January': 'Wet', 'February': 'Dry', 'March': 'Dry',
    'April': 'Dry', 'May': 'Dry', 'June': 'Wet', 'July': 'Wet',
    'August': 'Wet', 'September': 'Wet', 'October': 'Wet', 'November': 'Wet'
}

def get_season(month: str) -> str:
    return SEASON_MAP.get(month, "Wet")


#  GET REAL HISTORICAL FEATURES FROM FIRESTORE

def build_features_from_frontend(diseases, disease_case_counts, month, year):
    features = {}
    month_encoded = safe_transform(month_encoder, [month])[0]
    season_encoded = safe_transform(season_encoder, [get_season(month)])[0]

    if month_encoded == -1:
        month_encoded = 0
    if season_encoded == -1:
        season_encoded = season_encoder.transform(["Wet"])[0] if "Wet" in season_encoder.classes_ else 0

    for disease in diseases:
        disease_encoded = safe_transform(disease_encoder, [disease])[0]
        if disease_encoded == -1:
            continue

        # Get the current month's case count from frontend input
        current_cases = disease_case_counts.get(disease, 0)

        # Minimal feature input for the model
        features[disease] = [
            disease_encoded,      # Disease
            month_encoded,        # Month
            year,                 # Year
            season_encoded,       # Season
            current_cases,        # Use frontend current count as "prev cases"
            current_cases,        # Use frontend current count as rolling3
            0                     # TimeIndex (0 if unknown)
        ]

    return features

def fetch_historical_average(municipality, disease):
    try:
        healthworkers = db.collection("healthradarDB").document("users").collection("healthworker").get()
        counts = []
        for hw_doc in healthworkers:
            uploaded_cases = hw_doc.reference.collection("UploadedCases") \
                .where("Municipality", "==", municipality.strip().title()) \
                .where("DiseaseName", "==", disease.strip().title()) \
                .get()
            for case in uploaded_cases:
                counts.append(case.to_dict()["CaseCount"])
        return np.mean(counts) if counts else 0
    except:
        return 0



def safe_transform(encoder, values):
    """Transform values with LabelEncoder; assign -1 to unseen labels."""
    known_classes = set(encoder.classes_)
    transformed = []
    for v in values:
        if v in known_classes:
            transformed.append(encoder.transform([v])[0])
        else:
            transformed.append(-1)
    return np.array(transformed)

# Feature Engineering

def compute_features(records, target_month, target_year):
    df = pd.DataFrame(records)

    if df.empty:
        return None

    # Standardize disease names
    df["DiseaseName"] = (
        df["DiseaseName"]
        .str.strip()
        .str.title()
        .replace({
            "Sore Eys": "Sore Eyes",
            "Tubercolosis": "Tuberculosis",
            "Hiv": "HIV"
        })
    )

    # Use safe_transform for encoding
    df["Month_Encoded"] = safe_transform(month_encoder, df["Month"])
    df["Disease_Encoded"] = safe_transform(disease_encoder, df["DiseaseName"])
    
    # Season
    df["Season"] = df["Month"].map(SEASON_MAP)
    df["Season_Encoded"] = safe_transform(season_encoder, df["Season"])

    # Sort data
    df = df.sort_values(["DiseaseName", "Year", "Month_Encoded"]).reset_index(drop=True)

    # Calculate PrevMonthCases
    df["PrevMonthCases"] = (
        df.groupby("DiseaseName")["CaseCount"]
        .shift(1)
        .fillna(df["CaseCount"].mean())
    )

    # Calculate Rolling3
    df["Rolling3"] = (
        df.groupby("DiseaseName")["CaseCount"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # TimeIndex
    df["TimeIndex"] = df.groupby("DiseaseName").cumcount()

    # Build Input Features
    target_month_encoded = safe_transform(month_encoder, [target_month])[0]
    target_season_encoded = safe_transform(season_encoder, [get_season(target_month)])[0]
    
    if target_month_encoded == -1:
        target_month_encoded = 0
    
    if target_season_encoded == -1:
        target_season_encoded = season_encoder.transform(["Wet"])[0] if "Wet" in season_encoder.classes_ else 0

    prediction_inputs = {}
    unique_diseases = df["DiseaseName"].unique()
    
    for disease in unique_diseases:
        subset = df[df["DiseaseName"] == disease]
        
        if subset.empty:
            continue
            
        disease_encoded = safe_transform(disease_encoder, [disease])[0]
        if disease_encoded == -1:
            continue

        # Handle missing columns
        if "Rolling3" not in subset.columns or subset["Rolling3"].isna().any():
            rolling3 = subset["CaseCount"].tail(3).mean()
        else:
            rolling3 = subset.iloc[-1]["Rolling3"]
        
        prev_cases = subset.iloc[-1]["PrevMonthCases"]
        next_time_index = subset.iloc[-1]["TimeIndex"] + 1
        
        if pd.isna(rolling3):
            rolling3 = subset["CaseCount"].mean()
        
        if pd.isna(prev_cases):
            prev_cases = subset["CaseCount"].mean()

        prediction_inputs[disease] = [
            disease_encoded,
            target_month_encoded,
            target_year,
            target_season_encoded,
            prev_cases,
            rolling3,
            next_time_index
        ]

    return prediction_inputs if prediction_inputs else None


#  PREDICT USING REAL FEATURES - RETURN 1/0

@app.post("/predict_multiple")
def predict_multiple(request: MultiPredictionRequest):
    try:
        diseases = request.diseases
        disease_case_counts = request.disease_case_counts
        month = request.month
        year = request.year
        municipality = request.municipality

        # 1️⃣ Build features from frontend input
        inputs = build_features_from_frontend(diseases, disease_case_counts, month, year)

        if not inputs:
            return {"error": "No valid diseases to predict. Check the csv for pottential wrong spellings", "results": []}

        results = []
        for disease, feature_row in inputs.items():
            predicted_cases = model.predict([feature_row])[0]
            predicted_cases = max(0, round(predicted_cases))

            #  Get current month's average from Firestore for comparison
            current_cases = fetch_historical_average(municipality, disease)

            #  Determine rise (25% increase threshold)
            if current_cases > 0:
                increase_percentage = ((predicted_cases - current_cases) / current_cases) * 100
                prediction = 1 if increase_percentage >= 25 else 0
            else:
                prediction = 1 if predicted_cases > 0 else 0

            probability = min(100, max(0, predicted_cases))

            results.append({
                "disease_name": disease,
                "prediction": prediction,
                "probability": probability,
                "predicted_cases": predicted_cases,
                "current_cases": current_cases
            })

        results.sort(key=lambda x: x["probability"], reverse=True)
        rising_diseases = [r for r in results if r["prediction"] == 1]
        top_disease = rising_diseases[0]["disease_name"] if rising_diseases else None

        return {
            "target_month": month,
            "target_year": year,
            "results": results,
            "top_disease": top_disease,
            "rising_count": len(rising_diseases)
        }

    except Exception as e:
        print(f"Error in predict_multiple: {e}")
        return {"error": f"Internal server error: {str(e)}", "results": []}
