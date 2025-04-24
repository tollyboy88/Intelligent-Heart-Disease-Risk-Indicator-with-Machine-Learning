from flask import Flask, render_template, request, make_response
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import shutil
from datetime import datetime
from xhtml2pdf import pisa
from io import BytesIO

app = Flask(__name__)

# Directories
ARTIFACTS_DIR = 'flask_app_artifacts'
STATIC_DIR = 'static'

# Ensure static directory exists
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Copy plots from artifacts to static
for file in os.listdir(ARTIFACTS_DIR):
    if file.endswith('.png'):
        shutil.copy(os.path.join(ARTIFACTS_DIR, file), os.path.join(STATIC_DIR, file))

# Load model and encoders
model = joblib.load(os.path.join(ARTIFACTS_DIR, 'best_model_xgboost.joblib'))

label_encoders = {}
categorical_columns = [
    'Sex', 'GeneralHealth', 'LastCheckupTime', 'PhysicalActivities', 'RemovedTeeth',
    'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder',
    'HadKidneyDisease', 'HadArthritis', 'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
    'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands',
    'SmokerStatus', 'ECigaretteUsage', 'ChestScan', 'RaceEthnicityCategory', 'AgeCategory',
    'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'TetanusLast10Tdap',
    'HighRiskLastYear', 'CovidPos'
]
for col in categorical_columns:
    le = joblib.load(os.path.join(ARTIFACTS_DIR, f'label_encoder_{col}.joblib'))
    label_encoders[col] = le

feature_names = model.get_booster().feature_names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.to_dict()

    for col in categorical_columns:
        if col in user_input:
            if user_input[col] in label_encoders[col].classes_:
                user_input[col] = label_encoders[col].transform([user_input[col]])[0]
            else:
                user_input[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
        else:
            user_input[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]

    numerical_columns = ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'HeightInMeters', 'WeightInKilograms', 'BMI']
    for key in numerical_columns:
        if key in user_input:
            user_input[key] = float(user_input[key])
        else:
            user_input[key] = 0.0

    input_df = pd.DataFrame([user_input], columns=feature_names)

    prediction = model.predict(input_df)[0]
    prediction_prob = model.predict_proba(input_df)[0][1]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # SHAP Waterfall Plot
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=feature_names
        ),
        max_display=10,
        show=False
    )
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    shap_waterfall_path = os.path.join(STATIC_DIR, f'shap_waterfall_{timestamp}.png')
    plt.savefig(shap_waterfall_path, bbox_inches='tight', dpi=150)
    plt.close()

    top_features = np.argsort(np.abs(shap_values[0]))[-3:]
    text_explanation = f"Prediction: {'High Risk' if prediction == 1 else 'Low Risk'} (Probability: {prediction_prob:.2f})\n"
    for idx in top_features:
        feature_name = feature_names[idx]
        feature_value = input_df.iloc[0, idx]
        shap_value = shap_values[0][idx]
        text_explanation += f"Feature '{feature_name}' (value: {feature_value}) contributed {shap_value:.2f} to the prediction\n"

    shap_plots = {
        'SHAP Summary Plot': 'shap_summary_xgboost.png',
        'SHAP Waterfall Plot': f'shap_waterfall_{timestamp}.png'
    }

    fairness_plots = []
    for file in os.listdir(ARTIFACTS_DIR):
        if file.startswith('cm_'):
            if 'male' in file:
                fairness_plots.append(('Confusion Matrix - Men', file))
            elif 'female' in file:
                fairness_plots.append(('Confusion Matrix - Women', file))
            elif 'age' in file:
                age_category = file.replace('cm_old_', '').replace('.png', '')
                fairness_plots.append((f'Confusion Matrix - Age {age_category}', file))
            elif 'race' in file:
                race_category = file.replace('cm_white_', '').replace('.png', '')
                fairness_plots.append((f'Confusion Matrix - Race {race_category}', file))

    roc_plots = {
        'Overall': 'roc_curve_xgboost.png',
        'Sex': 'roc_sex.png',
        'Age': 'roc_age.png',
        'Race/Ethnicity': 'roc_race.png'
    }

    metrics_df = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'final_model_metrics.csv'))
    metrics = metrics_df.to_dict(orient='records')[0]

    return render_template('results.html',
                           prediction=prediction,
                           prediction_prob=prediction_prob,
                           shap_plots=shap_plots,
                           text_explanation=text_explanation,
                           fairness_plots=fairness_plots,
                           roc_plots=roc_plots,
                           metrics=metrics)

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    form = request.form

    prediction = form.get('prediction')
    prediction_prob = float(form.get('prediction_prob'))
    text_explanation = form.get('text_explanation')
    shap_plot = form.get('shap_plot')

    roc_plots = {}
    fairness_plots = []
    for key in form:
        if key.startswith('roc_'):
            roc_plots[key.replace('roc_', '').replace('_', ' ')] = form[key]
        elif key.startswith('fairness_'):
            fairness_plots.append((key.replace('fairness_', '').replace('_', ' '), form[key]))

    metrics = {}
    for key in form:
        if key.startswith('metric_'):
            metric_name = key.replace('metric_', '').replace('_', ' ')
            metrics[metric_name] = form[key]

    html = render_template('pdf_template.html',
                           prediction=prediction,
                           prediction_prob=prediction_prob,
                           text_explanation=text_explanation,
                           shap_plot=shap_plot,
                           roc_plots=roc_plots,
                           fairness_plots=fairness_plots,
                           metrics=metrics)

    pdf_file = BytesIO()
    pisa.CreatePDF(BytesIO(html.encode("utf-8")), dest=pdf_file)

    response = make_response(pdf_file.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'inline; filename=prediction_result.pdf'
    return response

if __name__ == '__main__':
    app.run(debug=True)
