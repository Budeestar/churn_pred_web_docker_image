from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)

model = pickle.load(open('model_rf.sav', 'rb'))

# Define a list of columns based on your provided structure
model_columns = [
    'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender_Female', 'gender_Male',
    'Partner_No', 'Partner_Yes', 'Dependents_No', 'Dependents_Yes', 
    'PhoneService_No', 'PhoneService_Yes', 'MultipleLines_No', 
    'MultipleLines_No phone service', 'MultipleLines_Yes', 
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_No', 'PaperlessBilling_Yes',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'tenure_group_1 - 12', 'tenure_group_13 - 24', 'tenure_group_25 - 36', 
    'tenure_group_37 - 48', 'tenure_group_49 - 60', 'tenure_group_61 - 72'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_input = request.get_json()
        input_df = pd.DataFrame([json_input])
        
        # Generating dummy variables for categorical features
        input_df_encoded = pd.get_dummies(input_df)
        print("got dummies")
        # Reindexing to match the training data structure, filling missing columns with 0
        input_df_encoded = input_df_encoded.reindex(columns=model_columns, fill_value=0)
        
        # Predicting with the model
        prediction = model.predict(input_df_encoded)
        print("prediction")
        probability = model.predict_proba(input_df_encoded)[:, 1]
        
        # Mapping prediction to labels if necessary
        prediction_label = ['Not Churned' if pred == 0 else 'Churned' for pred in prediction]
        print(prediction_label)
        return jsonify({
            'prediction': prediction_label,
            'probability': probability.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
