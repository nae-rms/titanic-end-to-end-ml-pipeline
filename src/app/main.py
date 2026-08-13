import sys
import os
from pathlib import Path
import pandas as pd
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

# Resolve absolute path to project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR / "src"))

from src.predict import load_model, predict_survival

# Load trained model
MODEL_PATH = BASE_DIR / "models" / "titanic_model.pkl"
model = load_model(str(MODEL_PATH))

# Initialize FastAPI
app = FastAPI(title="Titanic Model API")

# Define API request schema
class PassengerInput(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int
    FamilySize: int
    IsAlone: int

@app.get("/")
def root():
    return RedirectResponse(url="/gui")

@app.post("/predict")
def predict_api(passenger: PassengerInput):
    df = pd.DataFrame([passenger.model_dump()])
    prediction = predict_survival(model, df)[0]
    return {
        "survived": int(prediction),
        "status": "Survived" if prediction == 1 else "Not Survived"
    }

# Define Gradio logic
def predict_ui(pclass, sex, age, sibsp, parch, fare, embarked):
    sex_num = 0 if sex == "Male" else 1
    embarked_dict = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}
    embarked_num = embarked_dict.get(embarked, 2)
    
    family_size = int(sibsp) + int(parch) + 1
    is_alone = 1 if family_size == 1 else 0

    df = pd.DataFrame([{
        'Pclass': int(pclass),
        'Sex': sex_num,
        'Age': float(age),
        'SibSp': int(sibsp),
        'Parch': int(parch),
        'Fare': float(fare),
        'Embarked': embarked_num,
        'FamilySize': family_size,
        'IsAlone': is_alone
    }])

    pred = predict_survival(model, df)[0]
    return "🎉 Survived" if pred == 1 else "💀 Did Not Survive"

# Build UI
demo = gr.Interface(
    fn=predict_ui,
    inputs=[
        gr.Dropdown([1, 2, 3], label="Passenger Class (Pclass)", value=3),
        gr.Radio(["Male", "Female"], label="Sex", value="Male"),
        gr.Slider(0, 100, value=25, label="Age"),
        gr.Number(label="Siblings/Spouses Aboard (SibSp)", value=0),
        gr.Number(label="Parents/Children Aboard (Parch)", value=0),
        gr.Number(label="Fare Paid ($)", value=15.0),
        gr.Dropdown(["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"], label="Embarked Port", value="Southampton (S)"),
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Titanic Survival Predictor"
)

# Mount Gradio properly so config gets populated
app = gr.mount_gradio_app(app, demo, path="/gui")