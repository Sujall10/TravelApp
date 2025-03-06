import pandas as pd
import mysql.connector

db = mysql.connector.connect(
    host="localhost",       
    user="root",            
    password="root@123",    
    database="traveltrip"   
)

cursor = db.cursor()

# passenger_df = pd.read_excel('Datas\\PassengerFINALdataset.xlsx', engine='openpyxl')
flightss = pd.read_excel('Datas\\FlightFINALdataset.xlsx', engine='openpyxl')  # Replace with your flight dataset file path
# hotelss = pd.read_excel('Datas\\HotelFINALdataset.xlsx', engine='openpyxl') 
# guest_profile_df = pd.read_excel('Datas\\GuestFINALdataset.xlsx', engine='openpyxl') 
# car_rent_df = pd.read_excel('Datas\\CarFINALdataset.xlsx', engine='openpyxl')  # Replace with your file path
# merged_df = pd.read_excel('Datas\\MergedFINALdataset.xlsx', engine='openpyxl')
# review_df = pd.read_excel('Datas\\ReviewFINALdataset.xlsx', engine='openpyxl')

# car_rent_df['Check-in'] = pd.to_datetime(car_rent_df['Check-in'], format='%m/%d/%Y', errors='coerce').dt.strftime('%Y-%m-%d')

# merged_df['rentalDuration'].fillna('0',inplace=True)
# merged_df['Car_total_distance'].fillna('0',inplace=True)
# merged_df.fillna(value='None', inplace=True)


# merged_df.info()
# car_rent_df['Check-in'] = pd.to_datetime(car_rent_df['Check-in'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')

flightss['Departure_date'] = pd.to_datetime(flightss['Departure_date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')

# hotel_df['Check-in'] = pd.to_datetime(hotel_df['Check-in'], errors='coerce').dt.strftime('%Y-%m-%d')

# merged_df["Departure_date"] = pd.to_datetime(merged_df["Departure_date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
# merged_df["Check-in"] = pd.to_datetime(merged_df["Check-in"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")


# cursor.execute("SELECT user_id FROM user")
# existing_users = {row[0] for row in cursor.fetchall()} 

# for i, row in passenger_df.iterrows():
#     cursor.execute("""
#         INSERT INTO passenger ( usercode, company, name, gender)
#         VALUES ( %s, %s, %s, %s)
#     """, (row['User_ID'], row['company'], row['Name'], row['gender_x'] ))

# Iterate through the rows of the DataFrame and insert data into MySQL
for i, row in flightss.iterrows():
    cursor.execute("""
        INSERT INTO Test_flight (travelcode, user_id, departure, arrival, flight_type, 
                            flight_duration, flight_distance, flight_agency, departure_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (row['travelCode'], row['User_ID'], row['Departure'], row['Arrival'], row['flightType'], 
          row['Flight_duration'], row['Flight_Distance'], row['Flight_agency'], row['Departure_date']))

# Insert data into 'hotel' table
# for i, row in hotel_df.iterrows():
#     cursor.execute("""
#         INSERT INTO hotel (user_id, travel_code, hotel_name, arrival_place, hotel_stay, hotel_per_day_rent, 
#                            check_in, hotel_total_price)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
#     """, (row['User_ID'], row['travelCode'], row['Hotel_Name'], row['Arrival_place'], row['Hotel_stay'], 
#           row['Hotel_per_day_price'], row['Check-in'], row['Hotel_TotalPrice']))

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
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#     """, (row['User_ID'], row['travelCode'], row['Check-in'], row['pickupLocation'], row['dropoffLocation'], 
#           row['carType'], row['rentalAgency'], row['rentalDuration'], row['Car_total_distance'], 
#           row['fuelPolicy'], row['Car_bookingStatus'], row['total_rent_price']))

# car_rent_df.drop('Rent_Date',axis=1)



# for i, row in merged_df.iterrows():
#     cursor.execute("""
#         INSERT INTO Merged (User_ID, company, name, User_gender, User_age, travelcode, departure_place, 
#                             arrival_place, flight_type, flight_price, flight_duration, flight_distance, 
#                             flight_agency, departure_date, hotel_name, hotel_stay, hotel_price_per_day, 
#                             hotel_total_price, check_in, Pickup_Location, Dropoff_Location, car_type, 
#                             car_agency, car_rental_duration, Car_Total_Distance, Fuel_Policy, 
#                             Car_BookingStatus, car_total_price, total_trip_cost)
#         VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
#                 %s, %s, %s, %s, %s, %s, %s)
#     """, (row["User_ID"], row["company"], row["Name"], row["gender_passenger"], row["age_passenger"], 
#           row["travelCode"], row["Departure"], row["Arrival"], row["flightType"], 
#           row["Flight_price"], row["Flight_duration"], row["Flight_Distance"], row["Flight_agency"], 
#           row["Departure_date"], row["Hotel_Name"], row["Hotel_stay"], row["Hotel_per_day_price"], 
#           row["Hotel_TotalPrice"], row["Check-in"], row["Car_pickupLocation"], row["Car_dropoffLocation"], 
#           row["carType"], row["rentalAgency"], row["rentalDuration"], row["Car_total_distance"], 
#           row["fuelPolicy"], row["Car_bookingStatus"], row["total_rent_price"], row["total_trip_cost"]))

# merged_df = merged_df.astype(str).replace("nan", None)
# print(merged_df.head())  # Check if NaN values exist before inserting
# print(merged_df.head())
# review_df = review_df.where(pd.notna(review_df), None)
# review_df = review_df['overall_rating']
# review_df['overall_rating'] = pd.to_numeric(review_df['overall_rating'], errors='coerce').fillna(0).astype(int)


# for i, row in review_df.iterrows():
    # cursor.execute("""
        # INSERT INTO review (travelCode, User_ID,  review_car, review_hotel, review_flights, 
                            # flight_rating, hotel_rating, car_rating, overall_rating)
        # VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    # """, (row['travelCode'], row['User_ID'], row['review_car'], row['review_hotel'], 
        #   row['review_flights'], row['flight_rating'], row['hotel_rating'], row['car_rating'], row['overall_rating']))


db.commit()
cursor.close()
db.close()

print("Data imported successfully.")
