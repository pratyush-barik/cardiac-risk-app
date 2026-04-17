from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI()

# Load models
scaler = joblib.load("model/scaler.pkl")
lr = joblib.load("model/lr.pkl")
dt = joblib.load("model/dt.pkl")
rf = joblib.load("model/rf.pkl")
knn = joblib.load("model/knn.pkl")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Cardiac Risk Detection</title>
        <style>
            body {
                font-family: Arial;
                background-color: #f4f6f8;
                text-align: center;
            }

            .container {
                width: 420px;
                margin: auto;
                background: white;
                padding: 20px;
                margin-top: 30px;
                border-radius: 10px;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            }

            select, input {
                width: 100%;
                padding: 8px;
                margin-bottom: 12px;
            }

            button {
                width: 100%;
                padding: 10px;
                background: #007BFF;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 16px;
            }

            h2 {
                margin-bottom: 20px;
            }
        </style>
    </head>

    <body>
        <div class="container">
            <h2>Cardiac Risk Detection</h2>

            <form action="/predict" method="post">

                Age:
                <input type="number" name="Age" required>

                Chest Pain:
                <select name="Chest_Pain"><option value="1">Yes</option><option value="0">No</option></select>

                Shortness of Breath:
                <select name="Shortness_of_Breath"><option value="1">Yes</option><option value="0">No</option></select>

                Fatigue:
                <select name="Fatigue"><option value="1">Yes</option><option value="0">No</option></select>

                Palpitations:
                <select name="Palpitations"><option value="1">Yes</option><option value="0">No</option></select>

                Dizziness:
                <select name="Dizziness"><option value="1">Yes</option><option value="0">No</option></select>

                Swelling:
                <select name="Swelling"><option value="1">Yes</option><option value="0">No</option></select>

                Pain Arms/Jaw/Back:
                <select name="Pain_Arms_Jaw_Back"><option value="1">Yes</option><option value="0">No</option></select>

                Cold Sweats/Nausea:
                <select name="Cold_Sweats_Nausea"><option value="1">Yes</option><option value="0">No</option></select>

                High BP:
                <select name="High_BP"><option value="1">Yes</option><option value="0">No</option></select>

                High Cholesterol:
                <select name="High_Cholesterol"><option value="1">Yes</option><option value="0">No</option></select>

                Diabetes:
                <select name="Diabetes"><option value="1">Yes</option><option value="0">No</option></select>

                Smoking:
                <select name="Smoking"><option value="1">Yes</option><option value="0">No</option></select>

                Obesity:
                <select name="Obesity"><option value="1">Yes</option><option value="0">No</option></select>

                Sedentary Lifestyle:
                <select name="Sedentary_Lifestyle"><option value="1">Yes</option><option value="0">No</option></select>

                Family History:
                <select name="Family_History"><option value="1">Yes</option><option value="0">No</option></select>

                Chronic Stress:
                <select name="Chronic_Stress"><option value="1">Yes</option><option value="0">No</option></select>

                Gender:
                <select name="Gender"><option value="1">Male</option><option value="0">Female</option></select>

                <button type="submit">Predict</button>

            </form>
        </div>
    </body>
    </html>
    """


@app.post("/predict", response_class=HTMLResponse)
def predict(
    Age: int = Form(...),
    Chest_Pain: int = Form(...),
    Shortness_of_Breath: int = Form(...),
    Fatigue: int = Form(...),
    Palpitations: int = Form(...),
    Dizziness: int = Form(...),
    Swelling: int = Form(...),
    Pain_Arms_Jaw_Back: int = Form(...),
    Cold_Sweats_Nausea: int = Form(...),
    High_BP: int = Form(...),
    High_Cholesterol: int = Form(...),
    Diabetes: int = Form(...),
    Smoking: int = Form(...),
    Obesity: int = Form(...),
    Sedentary_Lifestyle: int = Form(...),
    Family_History: int = Form(...),
    Chronic_Stress: int = Form(...),
    Gender: int = Form(...)
):

    features = np.array([[
        Age, Chest_Pain, Shortness_of_Breath, Fatigue, Palpitations,
        Dizziness, Swelling, Pain_Arms_Jaw_Back, Cold_Sweats_Nausea,
        High_BP, High_Cholesterol, Diabetes, Smoking, Obesity,
        Sedentary_Lifestyle, Family_History, Chronic_Stress, Gender
    ]])

    features = scaler.transform(features)

    probs = [
        lr.predict_proba(features)[0][1],
        dt.predict_proba(features)[0][1],
        rf.predict_proba(features)[0][1],
        knn.predict_proba(features)[0][1]
    ]

    avg_prob = sum(probs) / len(probs)
    confidence = round(avg_prob * 100, 2)

    result = "⚠️ Heart Disease Risk Detected" if avg_prob >= 0.5 else "✅ No Risk"

    return f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial;
                background-color: #f4f6f8;
                text-align: center;
            }}

            .result-box {{
                width: 400px;
                margin: auto;
                margin-top: 100px;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            }}

            a {{
                display: inline-block;
                margin-top: 15px;
                text-decoration: none;
                color: #007BFF;
            }}
        </style>
    </head>

    <body>
        <div class="result-box">
            <h2>Prediction Result</h2>
            <h3>{result}</h3>
            <p><strong>Confidence:</strong> {confidence}%</p>
            <a href="/">Go Back</a>
        </div>
    </body>
    </html>
    """