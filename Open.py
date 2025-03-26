import requests

# OpenSky API URL
url = "https://opensky-network.org/api/states/all"

# Fetch data
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    
    # Extract flights for a specific departure or arrival airport
    DEP_IATA = "DEL"  # Example: Delhi
    ARR_IATA = "BOM"  # Example: Bangalore
    
    filtered_flights = [
        flight for flight in data.get("states", []) 
        if flight[2] == DEP_IATA or flight[2] == ARR_IATA
    ]

    if filtered_flights:
        print(f"✈️ Flights related to {DEP_IATA} and {ARR_IATA}:")
        for flight in filtered_flights:
            print(f"🆔 Flight ICAO: {flight[0]}")
            print(f"   ✈️ Aircraft: {flight[8]}")
            print(f"   🛫 Departure: {flight[2]}")
            print(f"   🛬 Arrival: {flight[3]}\n")
    else:
        print("❌ No flights found for the selected airports.")

else:
    print(f"❌ Error fetching flight data: {response.status_code}")
