import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'diabetes_model.pkl')

diabetes_model = pickle.load(open(MODEL_PATH, 'rb'))

def predict_diabetes(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = diabetes_model.predict(input_array)

    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"
