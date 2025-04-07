import joblib
from sqlalchemy import create_engine
import os 
import numpy as np 
import pandas as pd 
from dotenv import load_dotenv


preprocessor_Car = joblib.load("preprocessor_car.pkl")
model = joblib.load("best_price_prediction_model.pkl")
trained_feature_names = joblib.load("trained_feature_columns.pkl") 

load_dotenv()
DB_URL = os.getenv("DB_URL")

# Establishing database connection
engine = create_engine(DB_URL)
with engine.connect() as conn:
    Tst = pd.read_sql("SELECT * FROM Test_car", conn)
    car = pd.read_sql("SELECT * FROM Car___Table", conn)

# Merging datasets
query = pd.merge(Tst, car, on='Car_ID', how="inner")
print("Merged dataset preview:")
print(query.head())


def Feature(query):
    """Feature engineering function to create new variables before transformation."""
    query['Rental_Date'] = pd.to_datetime(query['Rental_Date'])
    query['Return_Date'] = pd.to_datetime(query['Return_Date'])
    
    # Time-Based Features
    query['Rental_Day'] = query['Rental_Date'].dt.dayofweek
    query['Weekend_Rental'] = (query['Rental_Day'] >= 5).astype(int)
    query['Rental_Hour'] = query['Rental_Date'].dt.hour
    query['Peak_Hour'] = query['Rental_Hour'].apply(lambda x: 1 if (7 <= x <= 10 or 17 <= x <= 20) else 0)
    query['Rental_Year'] = query['Rental_Date'].dt.year
    query['Rental_Month'] = query['Rental_Date'].dt.month
    
    # Seasonal Indicator
    query['Season'] = query['Rental_Month'].map({12: "Winter", 1: "Winter", 2: "Winter",
                                               3: "Spring", 4: "Spring", 5: "Spring",
                                               6: "Summer", 7: "Summer", 8: "Summer",
                                               9: "Fall", 10: "Fall", 11: "Fall"})
    #query = pd.get_dummies(query, columns=['Season'])
    
    query.drop(['Rental_ID', 'User_ID', 'Rental_Date', 'Return_Date', 'Rental_Month', 'Rental_Hour'], axis=1, inplace=True)
    
    query.fillna(query.select_dtypes(include=[np.number]).median(), inplace=True)
    
    return query

# Applying Feature Engineering
query = Feature(query)
query_transformed = preprocessor_Car.transform(query)
prediction = model.predict(query_transformed)
print("Predicted Car rental cost:",prediction)