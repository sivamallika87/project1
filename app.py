from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("models/svc.pkl", "rb"))

symptoms_dict = {
    "itching": 0,
    "skin_rash": 1,
    "joint_pain": 6,
    "fatigue": 14,
    "high_fever": 25,
}

diseases_list = {
    0: "Fungal infection",
    1: "Allergy",
    2: "GERD",
    3: "Chronic cholestasis",
    4: "Drug Reaction",
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        symptoms = request.form.get("symptoms").split(",")
        input_vector = np.zeros(len(symptoms_dict))
        for s in symptoms:
            s = s.strip()
            if s in symptoms_dict:
                input_vector[symptoms_dict[s]] = 1

        disease = diseases_list[model.predict([input_vector])[0]]
        return render_template("index.html", prediction=disease)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)