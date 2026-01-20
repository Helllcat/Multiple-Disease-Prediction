import pickle
import numpy as np
import os

# Get absolute path to models folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'parkinson_model.pkl')

parkinson_model = pickle.load(open(MODEL_PATH, 'rb'))

def predict_parkinson(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = parkinson_model.predict(input_array)

    if prediction[0] == 1:
        return "Parkinson's Detected"
    else:
        return "No Parkinson's"
