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