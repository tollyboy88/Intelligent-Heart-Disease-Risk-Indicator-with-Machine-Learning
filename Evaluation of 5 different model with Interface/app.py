from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
import os
import shap

app = Flask(__name__)

# Load preprocessing objects and model
model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('ordinal_encoder.pkl')

# Load the metrics table
metrics_df = pd.read_csv('static/final_model_metrics.csv')

# Load the SHAP text explanation
with open('static/shap_text_explanation_best_model.txt', 'r') as f:
    shap_text2 = f.read()

# Load the SHAP text explanation
with open('static/shap_text_explanation_best_model1.txt', 'r') as f:
    shap_text = f.read()

# Define exact feature order from training data
FEATURE_ORDER = [
    'Sex', 'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays',
    'LastCheckupTime', 'PhysicalActivities', 'SleepHours', 'RemovedTeeth',
    'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
    'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
    'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
    'DifficultyConcentrating', 'DifficultyWalking',
    'DifficultyDressingBathing', 'DifficultyErrands', 'SmokerStatus',
    'ECigaretteUsage', 'ChestScan', 'RaceEthnicityCategory',
    'HeightInMeters', 'WeightInKilograms', 'BMI', 'AlcoholDrinkers',
    'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'TetanusLast10Tdap',
    'HighRiskLastYear', 'CovidPos', 'Age'
]

# Categorical features (must match encoder's features)
CATEGORICAL_FEATURES = [
    'Sex', 'GeneralHealth', 'LastCheckupTime', 'PhysicalActivities',
    'RemovedTeeth', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer',
    'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
    'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
    'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing',
    'DifficultyErrands', 'SmokerStatus', 'ECigaretteUsage', 'ChestScan',
    'RaceEthnicityCategory', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12',
    'PneumoVaxEver', 'TetanusLast10Tdap', 'HighRiskLastYear', 'CovidPos'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # 1. Handle Age Category
        age_category = form_data.get('AgeCategory')
        if not age_category:
            return "Missing Age Category", 400
            
        age_mapping = {
            '18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37,
            '40-44': 42, '45-49': 47, '50-54': 52, '55-59': 57,
            '60-64': 62, '65-69': 67, '70-74': 72, '75-79': 77,
            '80+': 80
        }
        age = age_mapping.get(age_category, 50)

        # 2. Prepare features in exact training order
        features = []
        for feature in FEATURE_ORDER:
            if feature == 'Age':
                features.append(age)
                continue
                
            value = form_data.get(feature)
            if not value:
                return f"Missing required field: {feature}", 400
                
            # Handle data types
            if feature in CATEGORICAL_FEATURES:
                features.append(str(value))
            else:
                try:
                    features.append(float(value))
                except ValueError:
                    return f"Invalid value for {feature}: {value}", 400

        # 3. Separate categorical and numerical features
        categorical_values = [features[FEATURE_ORDER.index(col)] 
                            for col in CATEGORICAL_FEATURES]
        numerical_values = [features[FEATURE_ORDER.index(col)] 
                          for col in FEATURE_ORDER 
                          if col not in CATEGORICAL_FEATURES]

        # 4. Encode categorical features
        try:
            encoded_categorical = encoder.transform([categorical_values])[0]
        except ValueError as e:
            return f"Invalid categorical value: {str(e)}", 400

        # 5. Create final feature array
        final_features = np.concatenate([encoded_categorical, numerical_values])

        # 6. Scale features
        scaled_features = scaler.transform([final_features])

        # 7. Make prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]

        # 8. SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_features)

        metrics_html = metrics_df.to_html(index=False)

        

        return render_template('result.html',
                             prediction=prediction,
                             probability=probability,
                             shap_plot=shap_values,
                             shap_summary='shap_summary_best_model.png',
                             shap_force='shap_force_plot_best_model.png',
                             shap_text=shap_text,
                             shap_text2=shap_text2,
                             metrics_html=metrics_html,
                             fairness_curve='roc_sex.png',
                             roc_curves='roc_curves_all_models.png',
                             metrics_plot='model_metrics_comparison.png')

    except Exception as e:
        return f"Error processing request: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)