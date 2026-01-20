import streamlit as st
st.title("Disease Prediction")
st.write("Replace with your model and UI")
from diabetes import predict_diabetes
from heart import predict_heart_disease
from parkinson import predict_parkinson

# Page config
st.set_page_config(page_title="Multiple Disease Prediction System")

st.title("🩺 Multiple Disease Prediction System")

# Sidebar menu
disease = st.sidebar.selectbox(
    "Select Disease",
    ("Diabetes", "Heart Disease", "Parkinson's")
)

# ================= DIABETES =================
if disease == "Diabetes":
    st.subheader("Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose Level", 0, 200)
    bp = st.number_input("Blood Pressure", 0, 140)
    skin = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin Level", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
    age = st.number_input("Age", 1, 120)

    if st.button("Predict Diabetes"):
        result = predict_diabetes(
            [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
        )
        st.success(result)

# ================= HEART =================
elif disease == "Heart Disease":
    st.subheader("Heart Disease Prediction")

    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.number_input("Chest Pain Type", 0, 3)
    trestbps = st.number_input("Resting BP", 0, 200)
    chol = st.number_input("Cholesterol", 0, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.number_input("Rest ECG", 0, 2)
    thalach = st.number_input("Max Heart Rate", 0, 220)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0)
    slope = st.number_input("Slope", 0, 2)
    ca = st.number_input("CA", 0, 4)
    thal = st.number_input("Thal", 0, 3)

    if st.button("Predict Heart Disease"):
        result = predict_heart_disease(
            [age, sex, cp, trestbps, chol, fbs, restecg,
             thalach, exang, oldpeak, slope, ca, thal]
        )
        st.success(result)

# ================= PARKINSON =================
else:
    st.subheader("Parkinson's Disease Prediction")

    fo = st.number_input("MDVP:Fo(Hz)")
    fhi = st.number_input("MDVP:Fhi(Hz)")
    flo = st.number_input("MDVP:Flo(Hz)")
    jitter = st.number_input("Jitter(%)")
    shimmer = st.number_input("Shimmer")
    nhr = st.number_input("NHR")
    hnr = st.number_input("HNR")
    rpde = st.number_input("RPDE")
    dfa = st.number_input("DFA")
    spread1 = st.number_input("spread1")
    spread2 = st.number_input("spread2")
    d2 = st.number_input("D2")
    ppe = st.number_input("PPE")

    if st.button("Predict Parkinson's"):
        result = predict_parkinson(
            [fo, fhi, flo, jitter, shimmer, nhr, hnr,
             rpde, dfa, spread1, spread2, d2, ppe]
        )
        st.success(result)
