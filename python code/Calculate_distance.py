import requests
import math

def get_city_coordinates(city_name):
    """Fetches the coordinates of a given city using the GeoNames API."""
    username = 'tisha_agrawal_08'  # Replace with your GeoNames username
    url = f'http://api.geonames.org/searchJSON?q={city_name}&maxRows=1&username={username}'
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data['totalResultsCount'] > 0:
            city = data['geonames'][0]
            latitude = float(city['lat'])
            longitude = float(city['lng'])
            print(f"Coordinates of {city_name}: {latitude}, {longitude}")
            return latitude, longitude
        else:
            print(f"Could not find coordinates for {city_name}.")
    else:
        print(f"Error fetching city data: {response.status_code}")
    
    return None, None

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the great-circle distance between two points using the Haversine formula."""
    R = 6371  # Radius of Earth in km
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c  # Distance in km

def calculate_flight_duration(distance, speed=250):
    """Calculates estimated flight duration in hours (default speed: 900 km/h)."""
    duration = distance / speed  # Time = Distance / Speed
    hours = int(duration)
    minutes = int((duration - hours) * 60)
    return hours, minutes

if __name__ == "__main__":
    city1 = input("Enter the first city: ")
    city2 = input("Enter the second city: ")
    
    coord1 = get_city_coordinates(city1)
    coord2 = get_city_coordinates(city2)
    
    if coord1 and coord2 and None not in coord1 and None not in coord2:
        distance = haversine(coord1[0], coord1[1], coord2[0], coord2[1])
        hours, minutes = calculate_flight_duration(distance)
        
        print(f"Flight Distance: {distance:.2f} km")
        print(f"Estimated Flight Duration: {hours} hours {minutes} minutes")
    else:
        print("Could not fetch valid coordinates. Please try different city names.")
