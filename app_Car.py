from flask import Flask, request, jsonify
import numpy as np
from Inference_Car import predict_rent_price

app = Flask(__name__)

@app.route('/')
def home():
    return "Car Rent Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        predictions = predict_rent_price(data)
        return jsonify({'prediction': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
