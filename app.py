from flask import Flask, request, jsonify
import requests
import joblib
import numpy as np

app = Flask(__name__)

# URL of the model file in Google Drive
# sharable_link = "https://drive.google.com/file/d/16ouFjFu_l5OxevbydlWriTKC3rxh3zPJ/view?usp=sharing" that i copied

url = "https://drive.google.com/uc?id=16ouFjFu_l5OxevbydlWriTKC3rxh3zPJ"

def download_model(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

# Download and load the model
download_model(url, 'credit_card_classification_model.joblib')
model = joblib.load('credit_card_classification_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.get_json()
    
    # Ensure all required columns are in the input data
    required_columns = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
        'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
        'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age',
        'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
        'Credit_Score', 'Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan',
        'Home Equity Loan', 'Mortgage Loan', 'Not Specified', 'Payday Loan',
        'Personal Loan', 'Student Loan', 'Credit_Mix_Bad', 'Credit_Mix_Good',
        'Credit_Mix_Standard', 'Payment_of_Min_Amount_No', 'Payment_of_Min_Amount_Yes',
        'Payment_Behaviour_High_spent_Large_value_payments', 
        'Payment_Behaviour_High_spent_Medium_value_payments',
        'Payment_Behaviour_High_spent_Small_value_payments', 
        'Payment_Behaviour_Low_spent_Large_value_payments', 
        'Payment_Behaviour_Low_spent_Medium_value_payments', 
        'Payment_Behaviour_Low_spent_Small_value_payments',
        'Occupation_Encoded', 'Month_encoded'
    ]

    # Check if all required columns are present in the input
    for col in required_columns:
        if col not in data:
            return jsonify({'error': f'Missing required field: {col}'}), 400

    # Extract input features into a numpy array
    input_features = np.array([[
        data['Age'], data['Annual_Income'], data['Monthly_Inhand_Salary'], 
        data['Num_Bank_Accounts'], data['Num_Credit_Card'], data['Interest_Rate'], 
        data['Num_of_Loan'], data['Delay_from_due_date'], 
        data['Num_of_Delayed_Payment'], data['Changed_Credit_Limit'], 
        data['Num_Credit_Inquiries'], data['Outstanding_Debt'], 
        data['Credit_Utilization_Ratio'], data['Credit_History_Age'], 
        data['Total_EMI_per_month'], data['Amount_invested_monthly'], 
        data['Monthly_Balance'], data['Credit_Score'], data['Auto Loan'], 
        data['Credit-Builder Loan'], data['Debt Consolidation Loan'], 
        data['Home Equity Loan'], data['Mortgage Loan'], 
        data['Not Specified'], data['Payday Loan'], 
        data['Personal Loan'], data['Student Loan'], 
        data['Credit_Mix_Bad'], data['Credit_Mix_Good'], 
        data['Credit_Mix_Standard'], data['Payment_of_Min_Amount_No'], 
        data['Payment_of_Min_Amount_Yes'], 
        data['Payment_Behaviour_High_spent_Large_value_payments'], 
        data['Payment_Behaviour_High_spent_Medium_value_payments'], 
        data['Payment_Behaviour_High_spent_Small_value_payments'], 
        data['Payment_Behaviour_Low_spent_Large_value_payments'], 
        data['Payment_Behaviour_Low_spent_Medium_value_payments'], 
        data['Payment_Behaviour_Low_spent_Small_value_payments'], 
        data['Occupation_Encoded'], data['Month_encoded']
    ]])
    
    # Use the loaded model to make predictions
    prediction = model.predict(input_features)
    
    # Convert the prediction to a more readable format if needed
    prediction_result = prediction[0]  # Assuming prediction is a single value

    return jsonify({'prediction': prediction_result})

if __name__ == '__main__':
    app.run(debug=True)


# app = Flask(__name__)

# # sharable_link = "https://drive.google.com/file/d/16ouFjFu_l5OxevbydlWriTKC3rxh3zPJ/view?usp=sharing" that i copied

# url = "https://drive.google.com/uc?id=16ouFjFu_l5OxevbydlWriTKC3rxh3zPJ"


# def download_model(url, filename):
#     response = requests.get(url)
#     with open(filename, 'wb') as f:
#         f.write(response.content)
        
# download_model(url,'model.joblib')

# model = joblib.load('model.joblib') # load model 


    
