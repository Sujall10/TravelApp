import pandas as pd
import mysql.connector

# Establish database connection
try:
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root@123",
        database="traveltrip"
    )
    cursor = db.cursor()
    print("Connected to MySQL successfully!")
except mysql.connector.Error as err:
    print(f"Error: {err}")
    exit()

# Read the merged dataset
try:
    merged_df = pd.read_excel('Datas\\MergedFINALdataset.xlsx', engine='openpyxl')
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Handling missing values properly
merged_df['rentalDuration'].fillna(0, inplace=True)
merged_df['Car_total_distance'].fillna(0, inplace=True)

# Convert NaN to None for MySQL NULL values
merged_df = merged_df.where(pd.notna(merged_df), None)

# Ensure proper datetime formatting
date_columns = ["Departure_date", "Check-in"]
for col in date_columns:
    if col in merged_df.columns:
        merged_df[col] = pd.to_datetime(merged_df[col], errors="coerce")

# SQL Insert Query
insert_query = """
    INSERT INTO Merged (
        User_ID, company, name, User_gender, User_age, travelcode, departure_place, 
        arrival_place, flight_type, flight_price, flight_duration, flight_distance, 
        flight_agency, departure_date, hotel_name, hotel_stay, hotel_price_per_day, 
        hotel_total_price, check_in, Pickup_Location, Dropoff_Location, car_type, 
        car_agency, car_rental_duration, Car_Total_Distance, Fuel_Policy, 
        Car_BookingStatus, car_total_price, total_trip_cost
    ) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
            %s, %s, %s, %s, %s, %s, %s)
"""

# Convert DataFrame to list of tuples
data_to_insert = [
    (
        row["User_ID"], row["company"], row["Name"], row["gender_passenger"], row["age_passenger"], 
        row["travelCode"], row["Departure"], row["Arrival"], row["flightType"], 
        row["Flight_price"], row["Flight_duration"], row["Flight_Distance"], row["Flight_agency"], 
        row["Departure_date"], row["Hotel_Name"], row["Hotel_stay"], row["Hotel_per_day_price"], 
        row["Hotel_TotalPrice"], row["Check-in"], row["Car_pickupLocation"], row["Car_dropoffLocation"], 
        row["carType"], row["rentalAgency"], row["rentalDuration"], row["Car_total_distance"], 
        row["fuelPolicy"], row["Car_bookingStatus"], row["total_rent_price"], row["total_trip_cost"]
    )
    for _, row in merged_df.iterrows()
]

# Insert Data Efficiently
try:
    cursor.executemany(insert_query, data_to_insert)
    db.commit()
    print(f"{cursor.rowcount} records inserted successfully!")
except mysql.connector.Error as err:
    print(f"Error inserting data: {err}")
finally:
    cursor.close()
    db.close()
    print("Database connection closed.")