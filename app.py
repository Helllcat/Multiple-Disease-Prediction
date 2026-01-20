from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# --------------------------------------------------
# Load Models
# --------------------------------------------------
scaler = pickle.load(open("models/scaler.pkl", "rb"))
diabetes_model = pickle.load(open("models/diabetes_model.pkl", "rb"))
heart_model = pickle.load(open("models/heart_model.pkl", "rb"))
parkinson_model = pickle.load(open("models/parkinsons_model.pkl", "rb"))

app = FastAPI(
    title="Multi-Disease Prediction API",
    description="API for predicting Diabetes, Heart Disease & Parkinson's",
    version="1.0.0"
)

# --------------------------------------------------
# Input Schemas
# --------------------------------------------------

class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


class HeartInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


class ParkinsonInput(BaseModel):
    MDVP_Fo_Hz: float
    MDVP_Fhi_Hz: float
    MDVP_Flo_Hz: float
    Jitter_percent: float
    Shimmer: float


# --------------------------------------------------
# Prediction Functions
# --------------------------------------------------

def predict_diabetes(data):
    data = np.array(data).reshape(1, -1)
    return int(diabetes_model.predict(data)[0])


def predict_heart(data):
    data = scaler.transform(np.array(data).reshape(1, -1))
    return int(heart_model.predict(data)[0])


def predict_parkinson(data):
    data = scaler.transform(np.array(data).reshape(1, -1))
    return int(parkinson_model.predict(data)[0])


# --------------------------------------------------
# API Endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {"message": "Welcome to Multi-Disease Prediction API!"}


@app.post("/predict/diabetes")
def predict_diabetes_api(data: DiabetesInput):

    input_data = [
        data.Pregnancies, data.Glucose, data.BloodPressure,
        data.SkinThickness, data.Insulin, data.BMI,
        data.DiabetesPedigreeFunction, data.Age
    ]

    result = predict_diabetes(input_data)

    return {
        "disease": "Diabetes",
        "prediction": "Positive" if result == 1 else "Negative",
        "status": result
    }


@app.post("/predict/heart")
def predict_heart_api(data: HeartInput):

    input_data = [
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]

    result = predict_heart(input_data)

    return {
        "disease": "Heart Disease",
        "prediction": "Positive" if result == 1 else "Negative",
        "status": result
    }


@app.post("/predict/parkinson")
def predict_parkinson_api(data: ParkinsonInput):

    input_data = [
        data.MDVP_Fo_Hz, data.MDVP_Fhi_Hz, data.MDVP_Flo_Hz,
        data.Jitter_percent, data.Shimmer
    ]

    result = predict_parkinson(input_data)

    return {
        "disease": "Parkinson’s Disease",
        "prediction": "Positive" if result == 1 else "Negative",
        "status": result
    }
