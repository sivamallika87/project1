import numpy as np
import pickle

# Load model
model = pickle.load(open("models/svc.pkl", "rb"))

# Symptoms dictionary
symptoms_dict = {
    "itching": 0,
    "skin_rash": 1,
    "nodal_skin_eruptions": 2,
    "continuous_sneezing": 3,
    "joint_pain": 6,
    "fatigue": 14,
    "high_fever": 25,
    # (add remaining symptoms here)
}

diseases_list = {
    0: "Fungal infection",
    1: "Allergy",
    2: "GERD",
    3: "Chronic cholestasis",
    4: "Drug Reaction",
    # (complete list)
}

def predict_disease(user_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in user_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    prediction = model.predict([input_vector])[0]
    return diseases_list[prediction]


# Example
symptoms = ["itching", "skin_rash"]
print("Predicted Disease:", predict_disease(symptoms))