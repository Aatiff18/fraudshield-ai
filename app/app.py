from flask import Flask, render_template, request
import numpy as np
import joblib
import random

app = Flask(__name__)

model = joblib.load('../model/fraud_model.pkl')

@app.route('/')
def home():

    return render_template(
        'index.html',
        prediction_text="System Ready",
        score=0,
        color="white"
    )

@app.route('/predict', methods=['POST'])
def predict():

    amount = float(request.form['amount'])

    time = float(request.form['time'])

    features = np.zeros((1,30))

    features[0][0] = time
    features[0][29] = amount

    prediction = model.predict(features)

    if prediction[0] == 1:

        result = "⚠️ Fraudulent Transaction Detected"

        score = random.randint(70,99)

        color = "red"

    else:

        result = "✅ Secure Transaction"

        score = random.randint(1,30)

        color = "lime"

    return render_template(
        'index.html',
        prediction_text=result,
        score=score,
        color=color
    )

if __name__ == "__main__":
    app.run(debug=True)