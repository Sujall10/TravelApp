import pandas as pd
import mysql.connector

db = mysql.connector.connect(
    host="localhost",       
    user="root",            
    password="root@123",    
    database="travelapplication"   
)

cursor = db.cursor()

Flight_RoundTrip = pd.read_csv('notebooks\\flight_positive_reviews.csv') 

for i, row in Flight_RoundTrip.iterrows():
    cursor.execute("""
        INSERT INTO Positive_Reviews (Review)
        VALUES (%s)
    """, (row['positive reviews']))


db.commit()
cursor.close()
db.close()

print("Data imported successfully.")