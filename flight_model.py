import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
import optuna
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix

# Load environment variables
load_dotenv()
DB_URL = os.getenv("DB_URL")
engine = create_engine(DB_URL)

# Load Data
df_flight = pd.read_sql("SELECT * FROM flight", engine)
df_user = pd.read_sql("SELECT * FROM passenger", engine)

df_user.rename(columns={'usercode': 'user_id'}, inplace=True)

df = pd.merge(df_user, df_flight, on='user_id', how='inner')
df.drop(columns=['name'], inplace=True)

# Feature Engineering
df['flight_date'] = pd.to_datetime(df['departure_date'])
df['is_weekend_flight'] = (df['flight_date'].dt.weekday >= 5).astype(int)
df['flight_month'] = df['flight_date'].dt.month

df = df.drop(columns=['travelcode','user_id'])
X = df.drop(columns=['flight_price'])
y = df['flight_price']

print(df.head())
# Preprocessing Pipeline
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# Train-Test Split
X_transformed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Hyperparameter Optimization
def objective(trial):
    model = XGBRegressor(
        n_estimators=trial.suggest_int('n_estimators', 100, 500),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
best_params = study.best_params

# Train Best Model
best_xgb = XGBRegressor(**best_params)
best_xgb.fit(X_train, y_train)
y_pred_xgb = best_xgb.predict(X_test)

# Stacking Model
stacked_model = StackingRegressor(
    estimators=[('xgb', best_xgb)],
    final_estimator=XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
)
stacked_model.fit(X_train, y_train)
y_pred_stack = stacked_model.predict(X_test)

# Neural Network Model
def build_nn():
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['mae'])
    return model

nn_model = build_nn()
nn_callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]
nn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=1, callbacks=nn_callbacks)

y_pred_nn = nn_model.predict(X_test).flatten()

# Model Evaluation
def evaluate_model(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # classification__report = classification_report(y_true,y_pred)
    # confusion__matrix = confusion_matrix(y_true,y_pred)
    print(f"\n {name} Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
evaluate_model("Optimized XGBoost", y_test, y_pred_xgb)
evaluate_model("Stacking Model", y_test, y_pred_stack)
evaluate_model("Neural Network", y_test, y_pred_nn)

