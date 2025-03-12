from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import Flask-CORS
import pickle
import numpy as np
import os
import urllib.request
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Now you can access them
model_path = os.getenv('MODEL_PATH')
secret_key = os.getenv('SECRET_KEY')
debug = os.getenv('DEBUG', 'False')  # If DEBUG is not set, default to 'False'

# Optional: Convert 'debug' to a boolean
debug = debug.lower() == 'true'  # Convert to boolean (True if 'True', else False)

print(f"Model path: {model_path}")
print(f"Secret Key: {secret_key}")
print(f"Debug mode: {debug}")

# Function to download model and scaler from S3 URLs if running on Heroku
# Add the correct URL for the scaler
scaler_url = 'https://loan-model-bucket.s3.eu-north-1.amazonaws.com/standard_scaler.pkl'

# Function to download the model and scaler from S3 URLs
def download_file_from_s3(url, filename):
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename} from {url}")
    except Exception as e:
        print(f"Error downloading {filename} from {url}: {e}")

# Check if model_path is set to the S3 URL and download model
if model_path:
    download_file_from_s3(model_path, 'logistic_model.pkl')
else:
    print("Error: Model path is not set in environment variables.")

# Check if scaler URL is set and download scaler
if scaler_url:
    download_file_from_s3(scaler_url, 'standard_scaler.pkl')  # Use the correct scaler URL
else:
    print("Error: Scaler URL is not set.")


# Initialize the Flask app
app = Flask(__name__)

# Enable CORS for the app
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

# Load the pickled Logistic Regression model
with open('logistic_model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

# Load the StandardScaler used during model training
with open('standard_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

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

        # Extract features from the JSON and log them for debugging
        features = np.array(data['features']).reshape(1, -1)
        print(f"Features received for prediction: {features}")

        # Apply the same scaling that was applied during training
        features_scaled = scaler.transform(features)

        # Make predictions
        prediction = clf.predict(features_scaled)[0]
        prediction_proba = clf.predict_proba(features_scaled)[0].tolist()

        # Determine loan approval status
        if prediction == 1:
            loan_status = "Loan Approved"
            explanation = "The model predicts loan approval with a high probability."
        else:
            loan_status = "Loan Rejected"
            explanation = (
                "The model predicts loan rejection with a high probability. "
                "This may be due to insufficient credit score, low income, or high debt ratio."
            )

        # Return prediction and probabilities
        return jsonify({
            'loan_status': loan_status,
            'prediction': int(prediction),
            'probabilities': prediction_proba,
            'explanation': explanation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
import os

port = int(os.environ.get('PORT', 5000))  # Default to 5000 for local development
app.run(host='0.0.0.0', port=port)


import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
