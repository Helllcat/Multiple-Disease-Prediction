import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'heart_model.pkl')

heart_model = pickle.load(open(MODEL_PATH, 'rb'))

def predict_heart_disease(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = heart_model.predict(input_array)

    return "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
