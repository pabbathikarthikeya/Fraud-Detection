from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained fraud detection model
model = joblib.load("fraud_model.pkl")

@app.route("/")
def home():
    return "Fraud Detection API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features from the JSON request
        features = np.array(data["features"]).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(features)

        # Return the prediction as JSON
        return jsonify({"fraud_prediction": int(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
