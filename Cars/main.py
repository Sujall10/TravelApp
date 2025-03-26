import numpy as np
import pandas as pd
import mysql.connector

db = mysql.connector.connect(
    host="localhost",       
    user="root",            
    password="Tisha@08",    
    database="inbox"   
)

cursor = db.cursor()
# passenger_df = pd.read_excel('data\PassengerFINALdataset.xlsx', engine='openpyxl')
# flight_df = pd.read_excel('data\FlightFINALdataset.xlsx', engine='openpyxl')  # Replace with your flight dataset file path
# hotel_df = pd.read_excel('data\HotelFINALdataset.xlsx', engine='openpyxl') 
# guest_profile_df = pd.read_excel('data\GuestFINALdataset.xlsx', engine='openpyxl') 
# car_rent_df = pd.read_excel('data\CarFINALdataset.xlsx', engine='openpyxl')  # Replace with your file path
# Reviewss = pd.read_excel('data//Car_FR.xlsx')  # Replace with your file path
Test_car_df = pd.read_csv('data\CAR_RENTAL.csv')
# location_table_df = pd.read_excel('data//Location_Table (1).xlsx', engine='openpyxl')
# car_table_df = pd.read_excel('data//Car-----Table.xlsx', engine='openpyxl')
# rental_table_df = pd.read_csv('data\CAR_RENTAL.csv')
# isha = pd.read_excel("data\Location_Table (1).xlsx")
# Test_car_df['Check-in'] = pd.to_datetime(Test_car_df['Check-in'], format='%m/%d/%Y', errors='coerce').dt.strftime('%Y-%m-%d')

# Convert 'Rental_Date' and 'Return_Date' to proper MySQL format
# rental_table_df['rental_date'] = pd.to_datetime(
#     rental_table_df['rental_date'], format='mixed', errors='coerce'
# ).dt.strftime("%Y-%m-%d %H:%M:%S")
# rental_table_df['return_date'] = pd.to_datetime(
#     rental_table_df['return_date'], format='mixed', errors='coerce'
# ).dt.strftime("%Y-%m-%d %H:%M:%S")

Test_car_df['rental_date'] = pd.to_datetime(
    Test_car_df['rental_date'], format='mixed', errors='coerce'
).dt.strftime("%Y-%m-%d %H:%M:%S")
Test_car_df['return_date'] = pd.to_datetime(
    Test_car_df['return_date'], format='mixed', errors='coerce'
).dt.strftime("%Y-%m-%d %H:%M:%S")
# rental_table_df['rental_date'] = pd.to_datetime(rental_table_df['rental_date'], format="%d-%m-%Y %H:%M").dt.strftime("%Y-%m-%d %H:%M:%S")
# rental_table_df['return_date'] = pd.to_datetime(rental_table_df['return_date'], format="%d-%m-%Y %H:%M").dt.strftime("%Y-%m-%d %H:%M:%S")

# car_rent_df['Check-in'] = pd.to_datetime(car_rent_df['Check-in'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')

# flight_df['Departure_date'] = pd.to_datetime(flight_df['Departure_date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')

# hotel_df['Check-in'] = pd.to_datetime(hotel_df['Check-in'], errors='coerce').dt.strftime('%Y-%m-%d')

# cursor.execute("SELECT user_id FROM user")
# existing_users = {row[0] for row in cursor.fetchall()} 

# for i, row in passenger_df.iterrows():
#     cursor.execute("""
#         INSERT INTO passenger1 ( usercode, company, name, gender)
#         VALUES ( %s, %s, %s, %s)
#     """, (row['User_ID'], row['company'], row['Name'], row['gender_x'] ))

# Iterate through the rows of the DataFrame and insert data into MySQL
# for i, row in flight_df.iterrows():
#     cursor.execute("""
#         INSERT INTO flight (travelcode, user_id, departure, arrival, flight_type, flight_price, 
#                             flight_duration, flight_distance, flight_agency, departure_date)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#     """, (row['travelCode'], row['User_ID'], row['Departure'], row['Arrival'], row['flightType'], 
#           row['Flight_price'], row['Flight_duration'], row['Flight_Distance'], row['Flight_agency'], row['Departure_date']))

# Insert data into 'hotel' table
# for i, row in hotel_df.iterrows():
#     cursor.execute("""
#         INSERT INTO hotel (user_id, travel_code, hotel_name, arrival_place, hotel_stay, hotel_per_day_rent, 
#                            check_in, hotel_total_price)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
# Insert data into 'guest_profile' table
# for i, row in guest_profile_df.iterrows():
#     cursor.execute("""
#         INSERT INTO guest_profile (Guest_Id, TravelCode, Guest_Name, Guest_Gender, Age, 
#                                    Guest_PhoneNo, Guest_Email, IdProof)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#     """, (row['Guest_ID'], row['travelCode'], row['Guest_name'], row['Guest_Gender'], row['Age'], 
#           row['Guest_PhoneNo'], row['Guest_email'], row['idProof']))
    
# for i, row in car_rent_df.iterrows():
#     cursor.execute("""
#         INSERT INTO car_rent (User_ID, TravelCode, Rent_Date, Pickup_Location, Dropoff_Location, Car_Type, 
#                               Rental_Agency, Rental_Duration, Car_Total_Distance, Fuel_Policy, Car_BookingStatus, 
#                               Total_Rent_Price)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s
#     """, (row['User_ID'], row['travelCode'], row['Hotel_Name'], row['Arrival_place'], row['Hotel_stay'], 
#           row['Hotel_per_day_price'], row['Check-in'], row['Hotel_TotalPrice']))
# , %s, %s)
#     """, (row['User_ID'], row['travelCode'], row['Check-in'], row['pickupLocation'], row['dropoffLocation'], 
#           row['carType'], row['rentalAgency'], row['rentalDuration'], row['Car_total_distance'], 
#           row['fuelPolicy'], row['Car_bookingStatus'], row['total_rent_price']))

# car_rent_df.drop('Rent_Date',axis=1)

# for i, row in Test_car_df.iterrows():
#     cursor.execute("""
#        INSERT INTO Test_car (User_ID, TravelCode, Rent_Date, Pickup_Location, Dropoff_Location, Car_Type, 
#                               Rental_Agency, Rental_Duration, Car_Total_Distance, Fuel_Policy, Car_BookingStatus, 
#                               Total_Rent_Price)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)

#     """, (row['User_ID'], row['travelCode'], row['Check-in'], row['pickupLocation'], row['dropoffLocation'], 
#           row['carType'], row['rentalAgency'], row['rentalDuration'], row['Car_total_distance'], 
#           row['fuelPolicy'], row['Car_bookingStatus'], row['total_rent_price']))

# Test_car_df.drop('Rent_Date',axis=1)

# print(car_table_df.info())
# car_table_df.columns = car_table_df.columns.str.strip()

# data = [
#     (row['Car_Id'], row['LocationID'], row['Make'], row['Model'], row['Year of Manufacture'], row['Car Type'], row['Price per Hour (INR)'], row['Mileage (km/l)'], row['City'], row['Agency_Name'], row['Base_Fare'], row['Occupancy'], row['Fuel Policy'], row['AC'], row['Transmission'], row['Luggage Capacity'], row['Unlimited Mileage'], row['Free Cancellation'], row['Ratings'])
#     for _, row in car_table_df.iterrows()
# ]

# cursor.executemany("""
#     INSERT INTO car___table (Car_ID, Location_ID, Make, Model, Year_of_Manufacture, CarType, Price_per_Hour, Mileage, City, Agency_Name, Base_Fare,Occupancy,Fuel_Policy,AC,Transmission,Luggage_Capacity,Unlimited_Mileage,Free_Cancellation,Ratings)
#     VALUES (%s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s, %s, %s, %s)
# """, data)
# for car rental table 
# print(rental_table_df.info())

data = [
    (row['Car_Id'], row['user_id'], row['Pickup_Location'], row['rental_date'], row['return_date'], row['Duration_Hours'], row['Formatted_Duration'])
    for _, row in Test_car_df.iterrows()
]

cursor.executemany("""
    INSERT INTO Test_car ( Car_ID, User_ID, Pickup_Location, Rental_Date, Return_Date, Duration_Hours,Duration)
    VALUES (%s, %s, %s, %s, %s,%s, %s)
""", data)
#  for location table
# data = [
#     (row['LocationID'], row['Name'], row['Address'], row['City'], row['Country'])
#     for _, row in location_table_df.iterrows()
# ]
# cursor.executemany("""
#     INSERT INTO Location_Table (Location_ID, Name, Address, City, Country)
#     VALUES (%s, %s, %s, %s, %s)
# """, data)


# Review_df = Review_df.where(pd.notna(Review_df), None)
# for i, row in Reviewss.iterrows():
#     cursor.execute("""
#         INSERT INTO Car_review (Review)
#         VALUES (%s)
#     """, ( row['Review_text'],))
    
db.commit()
cursor.close()
db.close()
print("Data imported successfully.")