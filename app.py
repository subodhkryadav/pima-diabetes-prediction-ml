from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load Model
model = joblib.load("pima_diabetes_predictor.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    user_input = {}

    if request.method == "POST":
        try:
            # Fetch values
            fields = [
                "Pregnancies", "Glucose", "BloodPressure",
                "SkinThickness", "Insulin", "BMI",
                "DiabetesPedigreeFunction", "Age"
            ]

            values = []
            for f in fields:
                val = float(request.form[f])
                values.append(val)
                user_input[f] = val

            # Predict
            prediction = model.predict([values])[0]

            if prediction == 1:
                result = "🔴 Patient is Diabetic"
            else:
                result = "🟢 Patient is Not Diabetic"

        except:
            result = "⚠️ Please enter valid numeric values."

    return render_template("index.html", result=result, user_input=user_input)


if __name__ == "__main__":
    app.run(debug=True)
