# import pandas as pd

# def load_data(past_flights_path, future_flights_path, sample_size=5000):
#     # Load past flights
#     df_past = pd.read_excel(past_flights_path)
    
#     # Load a sample of future flights for efficiency
#     df_future = pd.read_excel(future_flights_path, usecols=["Departure", "Arrival", "Flight_agency", "flightType", "Predicted_Price", "flight_duration"], nrows=sample_size)
    
#     return df_past, df_future

# def extract_user_preferences(df_past):
#     # Most preferred airlines
#     top_airlines = df_past["Flight_agency"].value_counts().head(3).index.tolist()
    
#     # Most common flight type
#     preferred_flight_type = df_past["flightType"].mode()[0]
    
#     # Common departure & arrival locations
#     top_departures = df_past["Departure"].value_counts().head(3).index.tolist()
#     top_arrivals = df_past["Arrival"].value_counts().head(3).index.tolist()
    
#     # Price range
#     price_min = df_past["Calculated_Flight_Price"].min()
#     price_max = df_past["Calculated_Flight_Price"].max()
    
#     return top_airlines, preferred_flight_type, top_departures, top_arrivals, price_min, price_max

# def recommend_flights(df_future, departure, arrival, flight_type, sort_by):
#     # Filter based on user inputs
#     filtered_flights = df_future[
#         (df_future["Departure"] == departure) &
#         (df_future["Arrival"] == arrival) &
#         (df_future["flightType"] == flight_type)
#     ]
    
#     # Sorting criteria
#     if sort_by == "cheapest":
#         recommended_flights = filtered_flights.sort_values(by=["Predicted_Price"])
#     elif sort_by == "fastest":
#         recommended_flights = filtered_flights.sort_values(by=["flight_duration"])
#     elif sort_by == "best":
#         recommended_flights = filtered_flights.sort_values(by=["Predicted_Price", "flight_duration"])
#     else:
#         print("Invalid sorting option. Showing default cheapest flights.")
#         recommended_flights = filtered_flights.sort_values(by=["Predicted_Price"])
    
#     return recommended_flights.head(5)

# def main():
#     past_flights_path = "notebooks//FinalFlightCTO.xlsx"
#     future_flights_path = "April2025_May2025FlightDatas.xlsx"
    
#     # Load data
#     df_past, df_future = load_data(past_flights_path, future_flights_path)
    
#     # User inputs
#     departure = input("Enter Departure Airport: ")
#     arrival = input("Enter Arrival Airport: ")
#     flight_type = input("Enter Flight Type (Economy/Business/First Class): ")
#     sort_by = input("Sort by (cheapest/fastest/best): ")
    
#     # Get recommendations
#     recommendations = recommend_flights(df_future, departure, arrival, flight_type, sort_by)
    
#     # Print recommendations
#     print(recommendations)

# if __name__ == "__main__":
#     main()

import pandas as pd

future_flights_path = "April2025_May2025FlightDatas.xlsx"
df_future = pd.read_excel("April2025_May2025FlightDatas.xlsx")

def fetch_flights(df_future, departure, arrival, flight_type):
    filtered_flights = df_future[
        (df_future["Departure"] == departure) &
        (df_future["Arrival"] == arrival) &
        (df_future["flightType"] == flight_type)
        ]
    return filtered_flights

def recommend_flights(filtered_flights, sort_by):
    # Sorting criteria
    if sort_by == "cheapest":
        recommended_flights = filtered_flights.sort_values(by=["Predicted_Price"])
    elif sort_by == "fastest":
        recommended_flights = filtered_flights.sort_values(by=["flight_duration"])
    elif sort_by == "best":
        recommended_flights = filtered_flights.sort_values(by=["Predicted_Price", "flight_duration"])
    else:
        print("Invalid sorting option. Showing default cheapest flights.")
        recommended_flights = filtered_flights.sort_values(by=["Predicted_Price"])

    return recommended_flights

# User inputs
departure = input("Enter Departure Airport: ")
arrival = input("Enter Arrival Airport: ")
flight_type = input("Enter Flight Type (Economy/Business/First Class): ")
sort_by = input("Sort by (cheapest/fastest/best): ")

# Get filtered flights
filtered_flights = fetch_flights(df_future, departure, arrival, flight_type)

# Get recommendations
recommended_flights = recommend_flights(filtered_flights, sort_by)
# Print recommendations
print(recommended_flights)
# print(filtered_flights.sort_values(by=["Predicted_Price"]))