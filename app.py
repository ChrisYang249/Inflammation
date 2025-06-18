import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request, send_file, redirect, url_for, flash, send_from_directory
import pandas as pd
import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.secret_key = os.environ.get('SECRET_KEY', 'default-dev-key')

# Mapping of model options
MODEL_MAP = {
    'IgA ASCA EU': {
        'pickle': 'IgAmodel.pkl',
        'target': 'IgA ASCA EU',
        'pred_col': 'IgA_ASCA_EU_Prediction'
    },
    'IgG ASCA EU': {
        'pickle': 'IgGmodel.pkl',
        'target': 'IgG ASCA EU',
        'pred_col': 'IgG_ASCA_EU_Prediction'
    },
    'OmpC. EU': {
        'pickle': 'OmpCmodel.pkl',
        'target': 'OmpC. EU',
        'pred_col': 'OmpC_EU_Prediction'
    },
    'Cbir1 EU': {
        'pickle': 'CbirCmodel.pkl',
        'target': 'Cbir1 EU',
        'pred_col': 'Cbir1_EU_Prediction'
    },
    'ANCA EU': {
        'pickle': 'Anca.pkl',
        'target': 'ANCA EU',
        'pred_col': 'ANCA_EU_Prediction'
    }
}

EDA_TARGETS = [
    'IgA ASCA EU',
    'IgG ASCA EU',
    'OmpC. EU',
    'Cbir1 EU',
    'ANCA EU'
]

@app.route('/')
def home():
    model_options = list(MODEL_MAP.keys())
    return render_template('index.html', model_options=model_options)

@app.route('/predict', methods=['POST'])
def predict():
    selected_model = request.form.get('model_select')
    if selected_model not in MODEL_MAP:
        flash('Invalid model selection.')
        return redirect(url_for('home'))
    model_info = MODEL_MAP[selected_model]
    model_path = model_info['pickle']
    pred_col = model_info['pred_col']
    # Load the selected model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    if 'csv_file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['csv_file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        df = pd.read_csv(file_path)
        # Drop all target columns and IDs
        drop_cols = [
            "IgA ASCA EU", "IgA ASCA Pos.", "IgG ASCA EU", "IgG ASCA Pos.", "ASCA Panel",
            "OmpC. EU", "OmpC Pos.", "Cbir1 EU", "Cbir1 Fla. Pos.", "ANCA EU", "ANCA Pos.",
            "serum_id", "participant_id", "sample_name"
        ]
        X = df.drop([col for col in drop_cols if col in df.columns], axis=1)
        # Predict
        predictions = model.predict(X)
        df[pred_col] = predictions
        output_filename = f"predictions_{pred_col}_{file.filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        df.to_csv(output_path, index=False)
        # Preview first 20 rows
        preview_cols = [pred_col] + [col for col in X.columns[:5]]  # Show prediction + first 5 features
        preview_df = df[preview_cols].head(20)
        return render_template('results.html',
                               filename=output_filename,
                               tables=[preview_df.to_html(classes='table table-dark table-striped', index=False)],
                               marker=selected_model)
    return redirect(url_for('home'))

@app.route('/eda', methods=['POST'])
def eda():
    if 'eda_csv_file' not in request.files:
        flash('No file part for EDA')
        return redirect(url_for('home'))
    file = request.files['eda_csv_file']
    if file.filename == '':
        flash('No selected file for EDA')
        return redirect(url_for('home'))
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        df = pd.read_csv(file_path)
        # Identify feature columns (exclude all targets and IDs)
        drop_cols = [
            "IgA ASCA EU", "IgA ASCA Pos.", "IgG ASCA EU", "IgG ASCA Pos.", "ASCA Panel",
            "OmpC. EU", "OmpC Pos.", "Cbir1 EU", "Cbir1 Fla. Pos.", "ANCA EU", "ANCA Pos.",
            "serum_id", "participant_id", "sample_name"
        ]
        feature_cols = [col for col in df.columns if col not in drop_cols]
        img_filenames = []
        static_uploads = os.path.join('static', 'uploads')
        os.makedirs(static_uploads, exist_ok=True)
        for target in EDA_TARGETS:
            if target in df.columns:
                correlations = df[feature_cols].corrwith(df[target]).dropna()
                # Select top 20 features by absolute correlation
                correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)[:20]
                plt.figure(figsize=(10, 6))
                sns.barplot(x=correlations.values, y=correlations.index, color='#7C0A02')
                plt.title(f'Correlation with {target}')
                plt.xlabel('Correlation')
                plt.ylabel('Feature')
                plt.tight_layout()
                img_filename = f'eda_corr_{target.replace(" ", "_").replace(".", "")}.png'
                img_path = os.path.join(static_uploads, img_filename)
                plt.savefig(img_path)
                plt.close()
                img_filenames.append(os.path.join('uploads', img_filename))
        return render_template('eda_results.html', eda_images=img_filenames, targets=EDA_TARGETS)
    return redirect(url_for('home'))

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/antibodies')
def antibodies_page():
    return render_template('antibodies.html')

if __name__ == '__main__':
    app.run(debug=True)