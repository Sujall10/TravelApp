from sqlalchemy import create_engine
import pandas as pd


db_url = "mysql+mysqlconnector://sidd:Marijuana%401!@localhost:3306/traveltrip"
engine = create_engine(db_url)

df = pd.read_sql('select * from hotel', engine)

print(df)