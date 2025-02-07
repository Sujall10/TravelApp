from dotenv import load_dotenv
import os
from sqlalchemy import create_engine
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split

# load enviroment variables
load_dotenv()
DB_url = os.getenv('DB_url')
engine=create_engine(DB_url)

df = pd.read_sql("SELECT * FROM car_rent",engine)
# df.drop(['Rent_ID','User_ID','TravelCode'],axis=1)

df['month'] = df['Rent_Date'].dt.month

X = df.drop(columns = ['Total_Rent_Price'])
y = df['Total_Rent_Price']

# preprocessor pipeline

num_features = X.select_dtypes(include=['int64','float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', parse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

X_transformed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

