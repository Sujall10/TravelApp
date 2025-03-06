import pandas as pd
import mysql.connector

db = mysql.connector.connect(
    host="localhost",       
    user="root",            
    password="root@123",    
    database="traveltrip"   
)

cursor = db.cursor()

inference_Car_df = pd.read_excel('Datas\\Test_car.xlsx', engine='openpyxl')

inference_Car_df['Check-in'] = pd.to_datetime(inference_Car_df['Check-in'], format='%m/%d/%Y').dt.strftime('%Y-%m-%d')

for i, row in inference_Car_df.iterrows():
    cursor.execute("""
        INSERT INTO Test_car (User_ID, TravelCode, Rent_Date, Pickup_Location, Dropoff_Location, Car_Type, 
                              Rental_Agency, Rental_Duration, Car_Total_Distance, Fuel_Policy, Car_BookingStatus)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (row['User_ID'], row['travelCode'], row['Check-in'], row['pickupLocation'], row['dropoffLocation'], 
          row['carType'], row['rentalAgency'], row['rentalDuration'], row['Car_total_distance'], 
          row['fuelPolicy'], row['Car_bookingStatus']))

# inference_Car_df.drop('Rent_Date',axis=1)

db.commit()
cursor.close()
db.close()

print("Data imported successfully.")