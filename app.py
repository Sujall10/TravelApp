from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from Inference_Car import predict_rent_price
from Inference_Flight import predict_price

app = Flask(__name__)
CORS(app)  # Allow all origins

@app.route('/')
def home():
    return "Car Rent & Flight Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debugging print

        if not data or 'service' not in data:
            return jsonify({'error': 'No data received or missing service type'})
        
        service_type = data.pop('service')  # Extract service type
        
        if service_type == 'car':
            predictions = predict_rent_price(data)
        elif service_type == 'flight':
            predictions = predict_price(data)
        else:
            return jsonify({'error': 'Invalid service type. Use "car" or "flight"'})
        
        print("Predictions:", predictions)  # Debugging print
        return jsonify({'prediction': predictions.tolist()})  # Ensure JSON response is valid
    except Exception as e:
        print("Error:", str(e))  # Debugging print
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

