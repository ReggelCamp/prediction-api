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
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import joblib
import traceback

app = FastAPI()

# ---------- LOAD MODEL + ENCODERS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "PKL-files")

model            = joblib.load(os.path.join(MODEL_PATH, "updated_disease_case_predictor.pkl"))
month_encoder    = joblib.load(os.path.join(MODEL_PATH, "updated_month_encoder.pkl"))
disease_encoder  = joblib.load(os.path.join(MODEL_PATH, "updated_disease_encoder.pkl"))
season_encoder   = joblib.load(os.path.join(MODEL_PATH, "updated_season_encoder.pkl")) 
# ---------- SEASON HELPER ----------
def get_season(month: str) -> str:
    month = month.lower()
    if month in ["december", "january", "february"]:
        return "Wet"
    if month in ["march", "april", "may"]:
        return "Dry"
    if month in ["june", "july", "august", "september", "october", "november"]:
        return "Wet"
    return "unknown"

# ---------- REQUEST SCHEMA ----------
class PredictRequest(BaseModel):
    diseases: List[str]
    disease_case_counts: Dict[str, float]   # 👈 per-disease totals from UI
    month: str
    year: int

# ---------- ENDPOINT ----------
@app.post("/predict_multiple")
def predict_multiple(req: PredictRequest):
    try:
        month_encoded = month_encoder.transform([req.month])[0]
        season      = get_season(req.month)
        season_encoded = season_encoder.transform([season])[0]

        results = []

        for disease_name in req.diseases:
            case_count = req.disease_case_counts.get(disease_name, 0)

            try:
                disease_encoded = disease_encoder.transform([disease_name])[0]
            except ValueError:
                return {"error": f"Unknown disease: {disease_name}"}

            # synthetic features (same logic as before)
            prev_month_cases = case_count * 0.9
            rolling3 = (case_count + prev_month_cases + case_count * 1.1) / 3
            time_index = 1

            input_row = np.array([[
                disease_encoded,
                month_encoded,
                req.year,
                season_encoded,
                prev_month_cases,
                rolling3,
                time_index
            ]])

            predicted_cases = float(model.predict(input_row)[0])
            is_rise = 1 if predicted_cases > case_count * 1.25 else 0

            results.append({
                "disease_name": disease_name,
                "predicted_cases": round(predicted_cases, 2),
                "prediction": is_rise
            })

        return {"results": results, "season": season}

    except Exception as e:
        print("❌ Internal error:", traceback.format_exc())
        return {"error": f"Internal Server Error: {str(e)}"}

#copy 2
# import os
# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List, Dict, Union
# import numpy as np
# import joblib
# import traceback

# app = FastAPI()

# # === Load encoders and model ===
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "PKL-files")

# model = joblib.load(os.path.join(MODEL_PATH, "new_disease_case_predictor.pkl"))
# month_encoder = joblib.load(os.path.join(MODEL_PATH, "new_month_encoder.pkl"))
# disease_encoder = joblib.load(os.path.join(MODEL_PATH, "new_disease_encoder.pkl"))
# season_encoder = joblib.load(os.path.join(MODEL_PATH, "season_encoder.pkl"))

# # === Helper function for season ===
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

# # === Pydantic model for request ===
# class PredictRequest(BaseModel):
#     diseases: List[str]  # List of disease names
#     disease_case_counts: Dict[str, Union[int, float]]  # Dictionary with disease names as keys and case counts as values (int or float)
#     month: str
#     year: Union[int, float]  # Year can be int or float

# # === Predict multiple diseases ===
# @app.post("/predict_multiple")
# def predict_multiple(req: PredictRequest):
#     try:
#         # Encode month and season
#         month_encoded = month_encoder.transform([req.month])[0]
#         season = get_season(req.month)
#         season_encoded = season_encoder.transform([season])[0]

#         results = []

#         for disease_name in req.diseases:
#             case_count = float(req.disease_case_counts.get(disease_name, 0.0))  # Ensure it's float, default to 0.0

#             try:
#                 disease_encoded = disease_encoder.transform([disease_name])[0]
#             except Exception:
#                 return {"error": f"Unknown disease: {disease_name}"}

#             # Simulate previous month and rolling values per disease
#             prev_month_cases = case_count * 0.9
#             rolling3 = (case_count + prev_month_cases + case_count * 1.1) / 3

#             time_index = 1  # Placeholder since no real time series provided

#             input_data = np.array([[
#                 disease_encoded,
#                 month_encoded,
#                 req.year,
#                 season_encoded,
#                 prev_month_cases,
#                 rolling3,
#                 time_index
#             ]])

#             predicted_cases = float(model.predict(input_data)[0])
#             is_rise = 1 if predicted_cases > case_count * 1.2 else 0

#             results.append({
#                 "disease_name": disease_name,
#                 "current_cases": case_count,
#                 "predicted_cases": round(predicted_cases, 2),
#                 "prediction": is_rise
#             })

#         return {"results": results, "season": season}

#     except Exception as e:
#         print("❌ Internal error:", traceback.format_exc())
#         return {"error": f"Internal Server Error: {str(e)}"}
