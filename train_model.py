import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("datasets/Training.csv")

X = data.drop("prognosis", axis=1)
y = data["prognosis"]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train SVC model
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save model
pickle.dump(model, open("models/svc.pkl", "wb"))
pickle.dump(le, open("models/label_encoder.pkl", "wb"))

print("Model saved successfully")