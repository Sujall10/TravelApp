import os
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from math import radians, sin, cos, sqrt, atan2

# Load environment variables
load_dotenv()
DB_URL = os.getenv("a")

# Load trained model
pipeline = joblib.load("Flights\\PKL folders\\price_prediction_pipeline.pkl")  

df_loc = pd.read_excel('notebooks\\AirportCoordinates.xlsx')
# Establish database connection
engine = create_engine(DB_URL)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

flight_speeds = {
    "Akasa Air": 820,
    "Air India": 840,
    "IndiGo": 830,
    "SpiceJet": 820,
    "AirAsia India": 810,
    "GoAir": 815
}
# flight_df["Distance_km"] = flight_df.apply(lambda row: haversine(row["Dep_Lat"], row["Dep_Lon"], row["Arr_Lat"], row["Arr_Lon"]), axis=1)



def convert_to_hhmm(duration_hr):
    hours = int(duration_hr)
    minutes = int((duration_hr - hours) * 60)
    return f"{hours:02d}:{minutes:02d}"


            

def departure_coordinates(flight_data):
    flight_df = flight_data.merge(df_loc, left_on="Departure", right_on="Name", how="left")
    flight_df = flight_df.rename(columns={"Latitude": "Dep_Lat", "Longitude": "Dep_Lon"})
    return flight_df
    

def Arrival_coordinates(flight_data):
    flight_df = flight_data.merge(df_loc, left_on="Arrival", right_on="Name", how="left")
    flight_df = flight_df.rename(columns={"Latitude": "Arr_Lat", "Longitude": "Arr_Lon"})
    return flight_df

def fetch_flight_data():
    """Fetch test flight data from the database."""
    try:
        with engine.connect() as conn:
            flight_data = pd.read_sql('SELECT * FROM Test_Flight', conn)

            flight_df = departure_coordinates(flight_data)
            flight_df = Arrival_coordinates(flight_df)

            flight_df = flight_df.drop(columns=['LocationID_x','Name_x','Address_x','City_x','LocationID_y','Name_y','Address_y','City_y'])

            flight_df["flight_distance"] = flight_df.apply(lambda row: haversine(row["Dep_Lat"], row["Dep_Lon"], row["Arr_Lat"], row["Arr_Lon"]), axis=1)

            flight_df["Speed_kmh"] = flight_df["Flight_agency"].map(flight_speeds)            
            flight_df["EstimatedDuration_hr"] = flight_df["flight_distance"] / flight_df["Speed_kmh"]
            flight_df["EstimatedDuration_HHMM"] = flight_df["EstimatedDuration_hr"].apply(convert_to_hhmm)

            flight_df['flight_duration'] = flight_df['EstimatedDuration_HHMM'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
            
            flight_df = flight_df.drop(columns={'EstimatedDuration_hr','Dep_Lat','Dep_Lon','Arr_Lat','Arr_Lon'})

        return flight_df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def predict_flight_price(data):
    """Uses trained pipeline to predict flight prices."""
    try:
        predictions = pipeline.predict(data)
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

        # flight_data.to_excel('April2025_May2025FlightDatas.xlsx', index=False)
        # from IPython.display import FileLink
        # FileLink('April2025_May2025FlightDatas.xlsx')
        
        print(flight_data)
    else:
        print("No flight data available.")

