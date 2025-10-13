from fastapi import FastAPI
import joblib
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np


# 1. Create app
app = FastAPI()

# 2. Add middleware (allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Set model path (your folder)
# MODEL_PATH = r"C:\Users\REGGEL\OneDrive\Desktop\mahine_learning"
# print("✅ Using model path:", MODEL_PATH)

# # 4. Load model & encoders
# model = joblib.load(os.path.join(MODEL_PATH, "new_disease_case_predictor.pkl"))
# month_encoder = joblib.load(os.path.join(MODEL_PATH, "new_month_encoder.pkl"))
# disease_encoder = joblib.load(os.path.join(MODEL_PATH, "new_disease_encoder.pkl"))
# season_encoder = joblib.load(os.path.join(MODEL_PATH, "season_encoder.pkl"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "PKL-files")

model = joblib.load(os.path.join(MODEL_PATH, "new_disease_case_predictor.pkl"))
month_encoder = joblib.load(os.path.join(MODEL_PATH, "new_month_encoder.pkl"))
disease_encoder = joblib.load(os.path.join(MODEL_PATH, "new_disease_encoder.pkl"))
season_encoder = joblib.load(os.path.join(MODEL_PATH, "season_encoder.pkl"))

# 5. Define month → season mapping
def get_season(month: str) -> str:
    month = month.lower()
    if month in ["december", "january", "february"]:
        return "Wet"     # Northeast monsoon / dry season
    elif month in ["march", "april", "may"]:
        return "Dry"      # Hot dry season
    elif month in ["june", "july", "august", "september", "october", "november"]:
        return "Wet"        # Rainy season
    else:
        return "unknown"

# 6. Prediction endpoint
# @app.get("/predict")
# def predict(disease_name: str, case_count: float, month: str, year: int):
#     try:
#         # Encode categorical inputs
#         disease_encoded = disease_encoder.transform([disease_name])[0]
#         month_encoded = month_encoder.transform([month])[0]

#         # Determine and encode season
#         season = get_season(month)
#         season_encoded = season_encoder.transform([season])[0]

#         # Simulate missing engineered features
#         # You can later replace these with real previous or rolling values
#         prev_month_cases = case_count  # temporary assumption
#         rolling3 = case_count          # temporary assumption
#         time_index = 1                 # default; could track per disease

#         # Combine all inputs in the same order as model training
#         input_data = np.array([[
#             disease_encoded,
#             month_encoded,
#             year,
#             season_encoded,
#             prev_month_cases,
#             rolling3,
#             time_index
#         ]])

#         # Predict outcome
#         prediction = model.predict(input_data)[0]

#         return {
#             "prediction": round(float(prediction), 2),
#             "season": season
#         }

#     except Exception as e:
#         return {"error": str(e)}

@app.get("/predict")
def predict(disease_name: str, case_count: float, month: str, year: int):
    try:
        disease_encoded = disease_encoder.transform([disease_name])[0]
        month_encoded = month_encoder.transform([month])[0]
        season = get_season(month)
        season_encoded = season_encoder.transform([season])[0]

        prev_month_cases = case_count
        rolling3 = case_count
        time_index = 1

        input_data = np.array([[
            disease_encoded,
            month_encoded,
            year,
            season_encoded,
            prev_month_cases,
            rolling3,
            time_index
        ]])

        predicted_cases = model.predict(input_data)[0]

        # 🔥 Binary classification: 1 = rise, 0 = no rise
        is_rise = 1 if predicted_cases > case_count else 0

        return {
            "prediction": int(is_rise),
            "predicted_cases": round(float(predicted_cases), 2),
            "season": season
        }

    except Exception as e:
        return {"error": str(e)}
