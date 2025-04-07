import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np
import optuna

df = pd.read_excel('Hotels\\Final_Hotel_data.xlsx')
# print(df)
# print(df.info())

a = df.drop(columns={'travelCode','User_ID','Departure','Arrival','Total Price per Night','Total Cost','Days of Stay','Check-Out','Number of Bedrooms'})
# print(a.info())
# print(a)
# print(a['Hotel'].nunique())
# print(a['Amenities'].nunique())

a['Check-in_month'] = a['Check-in'].dt.month
a['Check-in_Date'] = a['Check-in'].dt.strftime('%d').astype(int)
a['Total_person'] = a['Number of Adults'] + a['Number of Children']
# print(a.info())
# print(a)
a = a.drop(columns=['Check-in'])
X = a.drop(columns = ['Room Price per Night'])
y = a['Room Price per Night']

# print(X.info())
# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


num_features = X.select_dtypes(include=['int32','int64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.to_list()

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
]) if cat_features else 'passthrough'

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

print(X)
print(X.info())

# b = a.select_dtypes(include={'int32','int64'})

# corr_matrix = b.corr()

# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 8))

# # Create the heatmap
# sns.heatmap(corr_matrix, annot=True, fmt=".5f", cmap="coolwarm", linewidths=0.5)

# plt.show()

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
    }

    model = XGBRegressor(**params, random_state=42, eval_metric="rmse", n_jobs=-1)
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_score = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring="neg_root_mean_squared_error").mean()
    
    return -cv_score  

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best_params = study.best_params
print("Best Parameters:", best_params)

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(**best_params, random_state=42, eval_metric="rmse", n_jobs=-1))
])

final_pipeline.fit(X_train, y_train)

y_pred_test = final_pipeline.predict(X_test)
y_pred_train = final_pipeline.predict(X_train)

def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{name} Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

evaluate_model("Training Set", y_train, y_pred_train)
evaluate_model("Test Set", y_test, y_pred_test)