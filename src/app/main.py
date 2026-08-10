from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from predictor import predict_survival

app = FastAPI(
    title="Titatic Prediction Model",
    description="Titanic prediction model for learning deploying models",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok"}


