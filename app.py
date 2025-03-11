from gettext import install
from os import error
from flask import Flask, request, jsonify
import pickle
#from flask_cors import CORS
import json
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

#CORS(app)

# Load the pickled Logistic Regression model
with open('logistic_model.pkl', 'rb') as file:
    clf = pickle.load(file)

# Define the API route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON data with features and returns the prediction.
    """
    try:
        # Parse the input JSON
        data = request.get_json()

        # Ensure features are provided
        if not data or 'features' not in data:
            return jsonify({'error': 'Invalid input. Provide JSON with "features" key.'}), 400

        # Convert features to a NumPy array
        # Ensure the order of features matches the model's training data
        features = np.array(data['features']).reshape(1, -1) 

        # Make predictions
        prediction = clf.predict(features)[0]
        prediction_proba = clf.predict_proba(features)[0].tolist()

        # Return prediction and probabilities
        return jsonify({
            'prediction': int(prediction),
            'probabilities': prediction_proba
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)