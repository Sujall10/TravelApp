import joblib
import numpy as np
from Preprocessor_Car import preprocess_input

# Load the trained model
model = joblib.load("best_model_Car.pkl")

def predict_price():
    """
    Takes user input, preprocesses it, and predicts the flight price.
    """
    # Example of expected input format (adjust based on your actual preprocessing needs)
    data = {
        # "airline": input("Enter Airline: "),
        "source": input("Enter Pickup : "),
        "destination": input("Enter Destination : "),
        "Rent_date": input("Enter rent date(HH:MM 24hr format): "),
        "car_type": input("Enter Car Type (HH:MM 24hr format): "),
        "Rental": input("Enter agency: "),
        "Duration": input("Enter duration of rent: "),
        "total_distance": input("Enter Distance: "),
        "fuel": input("Enter fuel policy: "),
        "booking status": input("Enter booking status: ")
    
    }


    print("\nRaw input data:", data)  

    # Preprocess the input
    transformed_data = preprocess_input(data)  
    print("Transformed data for prediction:", transformed_data)

    # Make the prediction
    prediction = model.predict(transformed_data)

    print("\nPredicted Flight Price:", np.round(prediction, 2))

# Call the function to test
predict_price()
