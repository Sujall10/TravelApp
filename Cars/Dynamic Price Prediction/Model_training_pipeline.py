import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import joblib
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
import math
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# Loading environment variables
load_dotenv()
DB_url = os.getenv('DB_url')
engine = create_engine(DB_url)

# Loading data from MySQL
with engine.connect() as conn:
    Rent = pd.read_sql("SELECT * FROM Rentals_Table", conn)
    car = pd.read_sql("SELECT * FROM car___table", conn)

# Merge datasets
data = pd.merge(Rent, car, on='Car_ID', how='inner')

# Feature Engineering Function
def Feature(data):
    data['Rental_Date'] = pd.to_datetime(data['Rental_Date'])
    data['Return_Date'] = pd.to_datetime(data['Return_Date'])
    
    # Time-Based Features
    data['Rental_Day'] = data['Rental_Date'].dt.dayofweek
    data['Weekend_Rental'] = (data['Rental_Day'] >= 5).astype(int)
    data['Rental_Hour'] = data['Rental_Date'].dt.hour
    data['Peak_Hour'] = data['Rental_Hour'].apply(lambda x: 1 if (7 <= x <= 10 or 17 <= x <= 20) else 0)
    data['Rental_Year'] = data['Rental_Date'].dt.year
    data['Rental_Month'] = data['Rental_Date'].dt.month
    
    # Seasonal Indicator
    data['Season'] = data['Rental_Month'].map({12: "Winter", 1: "Winter", 2: "Winter",
                                               3: "Spring", 4: "Spring", 5: "Spring",
                                               6: "Summer", 7: "Summer", 8: "Summer",
                                               9: "Fall", 10: "Fall", 11: "Fall"})
    # data = pd.get_dummies(data, columns=['Season'])
    
    # Drop unnecessary columns
    data.drop(['Rental_ID', 'User_ID', 'Rental_Date', 'Return_Date', 'Rental_Month', 'Rental_Hour'], axis=1, inplace=True)
    
    # Fill missing values with median
    data.fillna(data.select_dtypes(include=[np.number]).median(), inplace=True)
    
    return data

# Apply Feature Engineering
data = Feature(data)

# Define X (features) and y (target variable)
X = data.drop(columns=['Total_Amount'])
y = data['Total_Amount']

# Identify feature types
numeric_features = X.select_dtypes(include=['int64', 'int32', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Define preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
#  Saving trained feature names to prevent mismatches
joblib.dump(X.columns.tolist(), "trained_feature_columns.pkl")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Columns in X_train before preprocessing:", X_train.columns)


scaler = StandardScaler()
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

joblib.dump(preprocessor, "preprocessor_car.pkl")
print(" Preprocessor fitted and saved successfully!")

# Objective Function for Hyperparameter Tuning
def objective(trial, model_name, X_train, y_train):
    params = {}
    if model_name == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.1, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            # 'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.01, 1.0),
            # 'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0),
            'random_state': 42,
            # 'n_jobs': -1
        }
        model = XGBRegressor(**params)
    
    scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    return -scores.mean()
                          
 
# Training and Evaluating Models
best_models = {}
model_performance = {}

for model_name in ['XGBoost']:
    print(f"Optimizing {model_name}...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, model_name, X_train, y_train), n_trials=40)
    
    best_params = study.best_params
    print(f"Best parameters for {model_name}: {best_params}")
    
    best_model = {
        'XGBoost': XGBRegressor
    }[model_name](**best_params)
    
    best_model.fit(X_train, y_train)
    best_models[model_name] = best_model

    # Evaluating model
    y_pred = best_model.predict(X_test)
    y = best_model.predict(X_train)

    r2_test = r2_score(y_test, y_pred)
    rmse_test = math.sqrt(mean_squared_error(y_test, y_pred))
    mae_test = mean_absolute_error(y_test, y_pred)

    r2_train = r2_score(y_train, y)
    rmse_train = math.sqrt(mean_squared_error(y_train, y))
    mae_train = mean_absolute_error(y_train, y)
    
    model_performance[model_name] = {'R² Score': r2_test, 'RMSE': rmse_test, 'MAE': mae_test}
    print(f"{model_name} - R²: {r2_test:.4f}, RMSE: {rmse_test:.2f}, MAE: {mae_test:.2f}")
    print(f"{model_name} - R²: {r2_train:.4f}, RMSE: {rmse_train:.2f}, MAE: {mae_train:.2f}")

# Meta Model (Stacking)
stacking_model = StackingRegressor(
    estimators=[('xgb', best_models['XGBoost'])],
    
    # estimators=[('rf', best_models['Random Forest']), ('gb', best_models['Gradient Boosting']), ('xgb', best_models['XGBoost'])],
    final_estimator=Ridge(alpha=1.0)
)
stacking_model.fit(X_train, y_train)
best_models['Stacking Regressor'] = stacking_model


y_pred = stacking_model.predict(X_test)
y = stacking_model.predict(X_train)

r2_test = r2_score(y_test, y_pred)
rmse_test = math.sqrt(mean_squared_error(y_test, y_pred))
mae_test = mean_absolute_error(y_test, y_pred)

r2_train = r2_score(y_train, y)
rmse_train = math.sqrt(mean_squared_error(y_train, y))
mae_train = mean_absolute_error(y_train, y)

# model_performance[model_name] = {'R² Score': r2, 'RMSE': rmse, 'MAE': mae}
print(f"stacking_test - R²: {r2_test:.4f}, RMSE: {rmse_test:.2f}, MAE: {mae_test:.2f}")
print(f"stacking_train - R²: {r2_train:.4f}, RMSE: {rmse_train:.2f}, MAE: {mae_train:.2f}")

best_model_name = max(model_performance, key=lambda name: model_performance[name]['R² Score'])
best_model = best_models[best_model_name]  # Ensure the best model is selected before saving
print(best_model)

# Savimg the best model
joblib.dump(best_model, 'best_price_prediction_model.pkl')
print(f"\nBest model ({best_model_name}) saved successfully!")
