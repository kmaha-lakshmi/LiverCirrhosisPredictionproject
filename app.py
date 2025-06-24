from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('rf_acc_68.pkl', 'rb'))
normalizer = pickle.load(open('normalizer.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting form values in the exact order used during training
        age = float(request.form['Age'])
        gender = float(request.form['Gender'])  # Expecting 1 for Male, 0 for Female
        tb = float(request.form['TB'])
        db = float(request.form['DB'])
        alkphos = float(request.form['Alkphos'])
        sgpt = float(request.form['Sgpt'])
        sgot = float(request.form['Sgot'])
        tp = float(request.form['TP'])
        alb = float(request.form['ALB'])
        ag_ratio = float(request.form['A/G Ratio'])

        # Prepare and normalize input
        features = [age, gender, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio]
        normalized = normalizer.transform([features])

        # Predict
        prediction = model.predict(normalized)[0]
        probability = model.predict_proba(normalized)[0][1]

        result_text = "Liver Cirrhosis Detected" if prediction == 1 else "No Cirrhosis Detected"
        confidence = round(probability * 100, 2)

        return render_template('result.html', result=result_text, confidence=confidence)

    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == '__main__':
    import os
port = int(os.environ.get('PORT', 10000))
app.run(host='0.0.0.0', port=port)
