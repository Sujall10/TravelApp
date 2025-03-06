import joblib
import numpy as np
from Preprocessor_Car import preprocess_input
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import pandas as pd

# Load the trained model
model = joblib.load("best_model_Car.pkl")

# Load environment variables
load_dotenv()
DB_URL = os.getenv("DB_URL")

# Establish database connection
engine = create_engine(DB_URL)

with engine.connect() as conn:
    inference_Car = pd.read_sql('SELECT * FROM Test_car', conn)

def predict_rent_price(data):
    transformed_data = preprocess_input(data)
    print("Transformed data for prediction:", transformed_data)  

    predictions = model.predict(transformed_data)
    return np.round(predictions, 2)

# Drop non-feature columns (like 'Actual_Price' if it exists)
feature_columns = inference_Car.drop(columns=["Actual_Price"], errors="ignore")

# Predict prices
inference_Car["Predicted_Price"] = predict_rent_price(feature_columns)

print(inference_Car)
# Display results
# print(inference_Car[["Predicted_Price"]])

# (Optional) Save results to a new table in the database
# with engine.connect() as conn:
    # inference_Car.to_sql("Test_car_predictions", conn, if_exists="replace", index=False)
