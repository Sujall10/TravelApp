import joblib
import numpy as np
from Preprocessor_Car import preprocess_input

# Load trained model
model = joblib.load("best_model_Car.pkl")

def predict_rent_price(data):
    transformed_data = preprocess_input(data)
    predictions = model.predict(transformed_data)
    return np.round(predictions, 2)  # Return rounded predictions
