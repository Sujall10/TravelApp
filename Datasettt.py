# import requests
# import pandas as pd
# import time

# # List of ICAO airport codes
# AIRPORT_CODES = [
#     'CJB', 'IXR', 'GOI', 'AMD', 'HYD', 'IXA', 'BOM', 'PNQ', 'VGA', 'GWL',
#     'BLR', 'GAU', 'PAB', 'RPR', 'PNY', 'IXW', 'ISK', 'GOX', 'IXD', 'BDQ',
#     'LUH', 'MAA', 'BBI', 'TIR', 'DED', 'SXR', 'DEL', 'HGI', 'VNS', 'RUP',
#     'CCU', 'PYG', 'SHL', 'ATQ', 'BHO', 'JAI', 'KUU', 'TRV', 'KTU', 'STV',
#     'UDR', 'DIU', 'IXC', 'SLV', 'IMF', 'HSS', 'VTZ', 'IDR', 'MYQ', 'PAT'
# ]

# # OpenSky API URL
# OPENSKY_URL = "https://opensky-network.org/api/states/all"

# def get_departing_flights(airport_code):
#     """Fetch departing flights from OpenSky API for a given airport."""
#     response = requests.get(OPENSKY_URL)
    
#     if response.status_code != 200:
#         print(f"Failed to fetch data for {airport_code}")
#         return None

#     data = response.json()
#     flights = []
    
#     # OpenSky returns 'states' where the 3rd index represents origin ICAO
#     for flight in data.get("states", []):
#         if flight[2] and flight[2].upper() == airport_code:
#             flight_info = {
#                 "icao24": flight[0],   # Unique aircraft identifier
#                 "callsign": flight[1],  # Flight number
#                 "origin_airport": flight[2],
#                 "longitude": flight[5],
#                 "latitude": flight[6],
#                 "altitude": flight[7],
#                 "velocity": flight[9]
#             }
#             flights.append(flight_info)

#     return flights

# if __name__ == "__main__":
#     all_flights = {}

#     for code in AIRPORT_CODES:
#         print(f"Fetching flights departing from {code}...")
#         flight_data = get_departing_flights(code)
#         if flight_data:
#             all_flights[code] = flight_data
#         time.sleep(2)  # Delay to prevent excessive API requests

#     # Convert results to DataFrame for better visualization
#     all_flights_df = []
#     for airport, flights in all_flights.items():
#         for flight in flights:
#             flight["airport"] = airport  # Add airport code to each row
#             all_flights_df.append(flight)

#     df = pd.DataFrame(all_flights_df)
#     print(df)

#     # Save to CSV
#     df.to_csv("departing_flights.csv", index=False)
#     print("\nFlight details saved to departing_flights.csv")

# https://flightxml.flightaware.com/json/FlightXML3/AirlineFlightSchedules
#     ?startDate=2025-03-30
#     &endDate=2025-03-31
#     &origin=DEL
#     &destination=JFK
#     &howMany=10
#     &offset=0


# import requests
# from datetime import datetime, timedelta

# # Your API key from AviationStack
# API_KEY = "023680026faec432106756ad9f10f948"

# # Set departure & arrival airports
# DEP_IATA = "DEL"  # Example: Delhi
# ARR_IATA = "BLR"  # Example: New York

# # Set the month & year
# YEAR = 2025
# MONTH = 3  # March

# # Loop through all days in the month
# start_date = datetime(YEAR, MONTH, 1)
# end_date = datetime(YEAR, MONTH + 1, 1) if MONTH < 12 else datetime(YEAR + 1, 1, 1)

# current_date = start_date
# while current_date < end_date:
#     flight_date = current_date.strftime("%Y-%m-%d")
#     print(f"\n🔹 Fetching direct flights for {flight_date}...\n")

#     # API request
#     url = f"https://api.aviationstack.com/v1/flights?access_key={API_KEY}&dep_iata={DEP_IATA}&arr_iata={ARR_IATA}&flight_date={flight_date}"
#     response = requests.get(url)

#     if response.status_code == 200:
#         data = response.json()

#         # Filter direct flights (stopovers == 0)
#         direct_flights = [flight for flight in data.get("data", []) if flight.get("stopovers", 1) == 0]

#         if direct_flights:
#             for flight in direct_flights:
#                 print(f"✈️ Flight: {flight['flight']['iata']} | Airline: {flight['airline']['name']}")
#                 print(f"   🛫 Departure: {flight['departure']['airport']} at {flight['departure']['scheduled']}")
#                 print(f"   🛬 Arrival: {flight['arrival']['airport']} at {flight['arrival']['scheduled']}\n")
#         else:
#             print("❌ No direct flights found.")

#     else:
#         print(f"❌ Error fetching flights for {flight_date}: {response.status_code}")

#     # Move to the next day
#     current_date += timedelta(days=1)


import requests

# Your API key from AviationStack
API_KEY = "023680026faec432106756ad9f10f948"

# Set departure & arrival airports
DEP_IATA = "DEL"  # Example: Delhi
ARR_IATA = "BOM"  # Example: Mumbai

# API request
url = f"https://api.aviationstack.com/v1/flights?access_key={API_KEY}&dep_iata={DEP_IATA}&arr_iata={ARR_IATA}"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    if "data" in data and data["data"]:  # Check if flights exist
        for flight in data["data"]:
            print(f"✈️ Flight: {flight['flight']['iata']} | Airline: {flight['airline']['name']}")
            print(f"   🛫 Departure: {flight['departure']['airport']} at {flight['departure']['scheduled']}")
            print(f"   🛬 Arrival: {flight['arrival']['airport']} at {flight['arrival']['scheduled']}")
            print(f"   ⏳ Stopovers: {flight.get('stopovers', 'Unknown')}\n")
    else:
        print("❌ No flights found.")

else:
    print(f"❌ Error fetching flights: {response.status_code}")

    # print(data)
    # direct_flights = [flight for flight in data.get("data", []) if flight.get("stopovers", 1) == 0]
    # stopover_flights = [flight for flight in data.get("data", []) if flight.get("stopovers", 0) > 0]

    
    # if direct_flights:
    #     print("Direct Flights:")
    #     for flight in direct_flights:
    #         print(f"Flight: {flight['flight']['iata']} | Airline: {flight['airline']['name']}")
    #         print(f"    Departure: {flight['departure']['airport']} at {flight['departure']['scheduled']}")
    #         print(f"    Arrival: {flight['arrival']['airport']} at {flight['arrival']['scheduled']}\n")
    # else:
    #     print("No direct flights found.")

    
    # if stopover_flights:
    #     print("\nStopover Flights:")
    #     for flight in stopover_flights:
    #         print(f"Flight: {flight['flight']['iata']} | Airline: {flight['airline']['name']}")
    #         print(f"   Departure: {flight['departure']['airport']} at {flight['departure']['scheduled']}")
    #         print(f"   Arrival: {flight['arrival']['airport']} at {flight['arrival']['scheduled']}")
    #         print(f"   Stopovers: {flight.get('stopovers', 'Unknown')}\n")
    # else:
    #     print("\nNo stopover flights found.")

# else:
#     print(f"Error fetching flights: {response.status_code}")
