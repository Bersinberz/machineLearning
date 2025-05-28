from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('linear_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        g1 = float(request.form['G1'])
        g2 = float(request.form['G2'])
        studytime = float(request.form['studytime'])
        failures = float(request.form['failures'])
        absences = float(request.form['absences'])
        goout = float(request.form['goout'])

        avg_grade = (g1 + g2) / 2
        engagement_score = studytime - goout

        features = np.array([[g1, g2, studytime, failures, absences, avg_grade, engagement_score]])
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction_text=f"Predicted Final Grade (G3): {prediction:.2f}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
