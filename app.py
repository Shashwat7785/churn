from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model and preprocessing objects
# Ensure these were saved during your training phase!
model = joblib.load('random_forest_model_undersampled.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('encoders.pkl')

# The exact order of features required by the model
FEATURE_ORDER = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'tenure', 
    'SeniorCitizen', 'TotalCharges', 'MultipleLines', 'InternetService', 
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Columns that need Label Encoding
CATEGORICAL_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
    'PaperlessBilling', 'PaymentMethod'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        input_data = {}

        # 1. Parse and format inputs
        for feature in FEATURE_ORDER:
            val = form_data.get(feature, '')
            
            # Handle Numerics
            if feature in ['tenure', 'SeniorCitizen']:
                input_data[feature] = int(val) if val else 0
            elif feature in ['MonthlyCharges', 'TotalCharges']:
                input_data[feature] = float(val) if val else 0.0
            # Handle Categoricals
            else:
                input_data[feature] = val

        # Convert to DataFrame
        df_input = pd.DataFrame([input_data])

        # 2. Apply Preprocessing (Encoders)
        for col in CATEGORICAL_COLS:
            # We use .transform() to apply the mappings learned during training
            df_input[col] = encoders[col].transform(df_input[col])

        # 3. Apply Preprocessing (Scaling)
        # Ensure the columns match the exact order the scaler was trained on
        scaled_features = scaler.transform(df_input[FEATURE_ORDER])

        # 4. Predict
        prediction = model.predict(scaled_features)[0]
        
        # Determine output text
        if prediction == 1:
            result_text = "High Risk of Churn"
            color_class = "danger"
        else:
            result_text = "Likely to Stay (No Churn)"
            color_class = "success"

        return render_template('index.html', prediction_text=result_text, color_class=color_class)

if __name__ == "__main__":
    app.run(debug=True)