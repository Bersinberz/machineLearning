# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[f'f{i}']) for i in range(11)]
        scaled = scaler.transform([features])
        prediction = model.predict(scaled)[0]
        result = "Approved" if prediction == 1 else "Rejected"
        return render_template('index.html', result=result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
