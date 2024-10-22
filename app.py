from flask import Flask, request, jsonify, render_template
import requests
import joblib
import numpy as np
import os

app = Flask(__name__)

# URL of the model file in Google Drive
# sharable_link = "https://drive.google.com/file/d/16ouFjFu_l5OxevbydlWriTKC3rxh3zPJ/view?usp=sharing" that i copied

url = "https://drive.google.com/uc?id=16ouFjFu_l5OxevbydlWriTKC3rxh3zPJ"

def download_model(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an error for bad responses
        with open(filename, 'wb') as f:
            f.write(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading model: {e}")
        raise

# Download and load the model
download_model(url, 'credit_card_classification_model.joblib')

# Check if the model file exists before loading
if os.path.exists('credit_card_classification_model.joblib'):
    model = joblib.load('credit_card_classification_model.joblib')
else:
    raise FileNotFoundError("Model file was not downloaded successfully.")

# Define columns based on your provided dataset
columns = [
    'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
    'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
    'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age',
    'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
     'Auto Loan', 'Credit-Builder Loan', 
    'Debt Consolidation Loan', 'Home Equity Loan', 
    'Mortgage Loan', 'Not Specified', 'Payday Loan',
    'Personal Loan', 'Student Loan', 
    'Credit_Mix_Bad', 'Credit_Mix_Good', 
    'Credit_Mix_Standard', 
    'Payment_of_Min_Amount_No', 
    'Payment_of_Min_Amount_Yes',
    'Payment_Behaviour_High_spent_Large_value_payments',
    'Payment_Behaviour_High_spent_Medium_value_payments',
    'Payment_Behaviour_High_spent_Small_value_payments',
    'Payment_Behaviour_Low_spent_Large_value_payments',
    'Payment_Behaviour_Low_spent_Medium_value_payments',
    'Payment_Behaviour_Low_spent_Small_value_payments',
    'Occupation_Encoded', 
    'Month_encoded'
]

# Home route to render the input form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and predict credit score classification
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data and convert to numeric
        input_data = [request.form.get(col) for col in columns]
        
        # Convert input data to float (or int as needed)
        try:
            input_data = [float(i) for i in input_data]  # Ensure all inputs are numeric
        except ValueError as e:
            return jsonify({'error': f'Invalid input data: {e}'}), 400
        
        input_data = np.array(input_data).reshape(1, -1)  # Reshape for model input
        
        # Use the loaded model to make predictions
        prediction = model.predict(input_data)

        # Interpret prediction result
        if prediction[0] == 0:
            result = "Poor"
        elif prediction[0] == 1:
            result = "Good"
        else:
            result = "Standard"  # Example outcome
        
        # Return the result template
        return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
    
