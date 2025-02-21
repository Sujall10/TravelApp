import joblib
import numpy as np
from Preprocessor_Flight import preprocess_input

model = joblib.load("best_model_Flight.pkl")

def predict_price(data):
    print("Raw input data:", data) 

    transformed_data = preprocess_input(data)
    print("Transformed data for prediction:", transformed_data)  

    predictions = model.predict(transformed_data)
    return np.round(predictions, 2)

