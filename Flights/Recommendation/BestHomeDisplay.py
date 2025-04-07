import pandas as pd

def load_data(past_flights_path, future_flights_path, sample_size=None):
    df_past = pd.read_excel(past_flights_path, engine='openpyxl')
    df_future = pd.read_excel(
        future_flights_path, 
        usecols=["Departure", "Arrival", "Flight_agency", "flightType", "Predicted_Price", "flight_duration"], 
        nrows=sample_size
    )   
    return df_past, df_future

def extract_user_preferences(df_past):
    # Most preferred airlines
    top_airlines = df_past["Flight_agency"].value_counts().head(3).index.tolist()
    
    # Most common flight type
    preferred_flight_type = df_past["flightType"].mode()[0]
    
    # Common departure & arrival locations
    top_departures = df_past["Departure"].value_counts().head(3).index.tolist()
    top_arrivals = df_past["Arrival"].value_counts().head(3).index.tolist()
    
    # Price range
    price_min = df_past["Calculated_Flight_Price"].min()
    price_max = df_past["Calculated_Flight_Price"].max()
    
    return top_airlines, preferred_flight_type, top_departures, top_arrivals, price_min, price_max

def recommend_flights(df_future, top_airlines, preferred_flight_type, top_departures, top_arrivals, price_min, price_max):
    # Filter based on user preferences
    filtered_flights = df_future[
        (df_future["Departure"].isin(top_departures)) &
        (df_future["Arrival"].isin(top_arrivals)) &
        (df_future["Flight_agency"].isin(top_airlines)) &
        (df_future["flightType"] == preferred_flight_type) &
        (df_future["Predicted_Price"] >= price_min) &
        (df_future["Predicted_Price"] <= price_max)
    ]
    
    # Rank by price and duration
    recommended_flights = filtered_flights.sort_values(by=["Predicted_Price", "flight_duration"]).head(5)
    
    return recommended_flights

def main():
    past_flights_path = "notebooks\\FinalFlightCTO.xlsx"
    future_flights_path = "April2025_May2025FlightDatas.xlsx"
    
    # Load data
    df_past, df_future = load_data(past_flights_path, future_flights_path)
    
    # Extract user preferences
    top_airlines, preferred_flight_type, top_departures, top_arrivals, price_min, price_max = extract_user_preferences(df_past)
    
    # Get recommendations
    recommendations = recommend_flights(df_future, top_airlines, preferred_flight_type, top_departures, top_arrivals, price_min, price_max)
    
    # Print recommendations
    print(recommendations)

if __name__ == "__main__":
    main()
