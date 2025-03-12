# Loan Prediction Web App

This project is a loan prediction web application built using **Flask** and **scikit-learn**. The application uses a Logistic Regression model to predict whether a loan application will be approved or rejected based on the applicant's data such as income, loan amount, CIBIL score, and other factors.

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Technologies Used](#technologies-used)
4. [How to Run](#how-to-run)
5. [Features](#features)
6. [License](#license)

## Overview

The **Loan Prediction Web App** takes user input from a web form, processes it, and provides a prediction on whether the loan will be approved or rejected. The prediction is based on a pre-trained **Logistic Regression** model and **StandardScaler** for scaling the input features. The project includes a front-end built with HTML and CSS and a backend using Flask.

## Project Structure

```plaintext
├── static/
│   └── css/
│       └── styles.css        # Custom CSS for styling the web app
├── templates/
│   └── index.html            # HTML file for the form and display of results
├── .gitignore                # Specifies files to ignore in the Git repository
├── app.py                    # Main Flask application script
├── app2.py                   # Additional app script for testing
├── FeatureEngineering.ipynb   # Jupyter Notebook for feature engineering
├── Hackathon_2024_Term2_LoanDatasetApproval.ipynb # Jupyter Notebook for the hackathon
├── PCA_Hackathon.ipynb       # PCA Analysis for feature reduction
├── LICENSE                   # Apache-2.0 License for the project
├── loan_approval_dataset.csv # Dataset used for training the model
├── logistic_model.pkl        # Trained Logistic Regression model
├── reduced_loan_data.csv     # Reduced dataset for feature selection
├── req.py                    # Required libraries file
├── req2.py                   # Additional required libraries file
└── standard_scaler.pkl       # Trained StandardScaler
```
## Technologies Used

- **Flask**: Web framework for building the web application.
- **scikit-learn**: Machine learning library for creating the logistic regression model.
- **HTML/CSS**: For building the web interface and styling.
- **Python**: Backend programming language.
- **Heroku**: For deploying the web app online.

## How to Run

Follow these steps to run the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Loan-Prediction-App.git
   
2. Navigate to the project directory:
   ```bash
    Copy
    cd Loan-Prediction-App
   
3. Create a virtual environment and activate it:
  For Windows:
   ```bash
      Copy
      python -m venv venv
      venv\Scripts\activate
   
  For macOS/Linux:
  
    ```bash
      Copy
      python3 -m venv venv
      source venv/bin/activate
        
 4. Install the required dependencies:
    ```bash
        Copy
        pip install -r requirements.txt

5. Set up environment variables:
You will need a .env file in the project directory with the following variables:

    ```plaintext
        Copy
        MODEL_PATH=your_model_url
        SECRET_KEY=your_secret_key
        DEBUG=True
        
6. Run the Flask app:

    ```bash
    Copy
    python app.py
7. Open the app in your browser:

Go to http://127.0.0.1:5000/

## Features
- Loan Prediction: Input data like income, loan amount, term, CIBIL score, and other features to predict if the loan will be approved or rejected.
- Result Display: View the status of the loan (Approved or Rejected) with probabilities.
- User-Friendly Interface: The app features a simple and easy-to-use interface with CSS for styling.
- Error Handling: The application handles missing or invalid input and provides relevant error messages.
  
## License
This project is licensed under the Apache-2.0 License - see the LICENSE file for details.
