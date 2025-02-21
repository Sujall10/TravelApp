import joblib

# Load Preprocessor
preprocessor = joblib.load("preprocessor_Car.pkl")

# Get feature names
try:
    preprocessor_features = preprocessor.get_feature_names_out()
except AttributeError:
    preprocessor_features = preprocessor.feature_names_in_  # Alternative for older versions

print("Preprocessor Features:", preprocessor_features)

# Load Model
model = joblib.load("best_model_Car.pkl")

# Check base models
for estimator_name, estimator in model.estimators_:
    print(f"Base Model: {estimator_name}")
    if hasattr(estimator, "feature_names_in_"):
        print("Feature Names:", estimator.feature_names_in_)

