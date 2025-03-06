import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from Preprocessor_Flight import preprocess_input  # Ensure this module properly preprocesses flight data

# Load the trained flight price prediction model
model = joblib.load("best_model_Flight.pkl")

# Load environment variables
load_dotenv()
DB_URL = os.getenv("DB_URL")

# Establish database connection
engine = create_engine(DB_URL)

def fetch_flight_data():
    """Fetches test flight data from the database."""
    try:
        with engine.connect() as conn:
            flight_data = pd.read_sql('SELECT * FROM Test_Flight', conn)
            df_user = pd.read_sql("SELECT * FROM passenger", conn)

        df_user.rename(columns={'usercode': 'user_id'}, inplace=True)
        df = pd.merge(df_user, flight_data, on='user_id', how='inner').drop(columns=['name'])

        return df
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def predict_flight_price(data):
    """Preprocesses input data and predicts flight prices."""
    try:
        transformed_data = preprocess_input(data)  # Ensure this function works for flight data
        print("Transformed data for prediction:", transformed_data)

        predictions = model.predict(transformed_data)
        return np.round(predictions, 2)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

if __name__ == "__main__":
    flight_data = fetch_flight_data()
    
    if flight_data is not None:
        # Drop any non-feature columns if present
        feature_columns = flight_data.drop(columns=["Actual_Price"], errors="ignore")

        # Predict prices
        flight_data["Predicted_Price"] = predict_flight_price(feature_columns)

        print(flight_data)
    else:
        print("No flight data available.")
