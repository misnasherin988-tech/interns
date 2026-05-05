from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.joblib")
columns = joblib.load("columns.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    input_data = {
        "Hours_Studied": float(request.form["Hours_Studied"]),
        "Attendance": float(request.form["Attendance"]),
        "Parental_Involvement": float(request.form["Parental_Involvement"]),
        "Access_to_Resources": float(request.form["Access_to_Resources"]),
        "Extracurricular_Activities": float(request.form["Extracurricular_Activities"]),
        "Sleep_Hours": float(request.form["Sleep_Hours"]),
        "Previous_Scores": float(request.form["Previous_Scores"]),
        "Motivation_Level": float(request.form["Motivation_Level"]),
        "Internet_Access": float(request.form["Internet_Access"]),
        "Tutoring_Sessions": float(request.form["Tutoring_Sessions"]),
        "Family_Income": float(request.form["Family_Income"]),
        "Teacher_Quality": float(request.form["Teacher_Quality"]),
        "School_Type": float(request.form["School_Type"]),
        "Peer_Influence": float(request.form["Peer_Influence"]),
        "Physical_Activity": float(request.form["Physical_Activity"]),
        "Learning_Disabilities": float(request.form["Learning_Disabilities"]),
        "Parental_Education_Level": float(request.form["Parental_Education_Level"]),
        "Distance_from_Home": float(request.form["Distance_from_Home"]),
        "Gender": float(request.form["Gender"])
    }

    df = pd.DataFrame([input_data])

    # Ensure correct order
    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)[0]

    return render_template("index.html",
                           prediction_text=f"Predicted Exam Score: {prediction:.2f}")

if __name__ == "__main__":
    app.run(debug=True)