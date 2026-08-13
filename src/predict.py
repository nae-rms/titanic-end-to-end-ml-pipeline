import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "titanic_model.pkl")

def load_model(model_path=MODEL_PATH):
    return joblib.load(model_path)

def predict_survival(model, passenger_data):
    return model.predict(passenger_data)
def main():
    model = load_model()
    print("Model loaded!")

    sample_passengers = pd.DataFrame({
        'Pclass': [1, 3, 2],
        'Sex': [0, 1, 0],    
        'Age': [25, 65, 35],
        'SibSp': [0, 0, 1],
        'Parch': [0, 0, 2],
        'Fare': [50, 8, 30],
        'Embarked': [2, 0, 1],
        'FamilySize': [1, 1, 4],
        'IsAlone': [1, 1, 0]
    })

    predictions = predict_survival(model, sample_passengers)

    print("Results:")
    for i, pred in enumerate(predictions):
        survival_status = "Survived" if pred == 1 else "Not Survived"
        print(f"Passenger {i+1}: {survival_status}")

if __name__ == "__main__":
    main()