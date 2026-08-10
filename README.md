# 🚢 Titanic End-to-End Machine Learning Pipeline

An end-to-end Machine Learning web application and REST API built to predict passenger survival on the Titanic based on demographic and voyage details.

## 📌 Project Overview

This project implements a complete ML pipeline—from model inference to web deployment—integrating a **FastAPI** backend, a **Gradio** interactive web UI, and a pre-trained **Scikit-Learn** predictive model.

* **Live Demo:** [Titanic Survival Predictor](https://titanic-end-to-end-ml-pipeline.onrender.com)
* **Web Interface:** Interactive Gradio UI accessible at `/gui`
* **REST API:** FastAPI backend supporting programmatic JSON requests at `/predict`

---

## 🛠️ Features

* **Interactive Web GUI:** Users can select passenger details (Class, Sex, Age, Fare, SibSp, Parch, Embarked Port) using intuitive controls and receive immediate predictions.
* **REST API Endpoint:** Clean `POST /predict` endpoint built with Pydantic schema validation for programmatic access.
* **Feature Engineering On-the-Fly:** Automatically calculates derived features (`FamilySize`, `IsAlone`) from raw user input prior to model inference.

---

## 🚀 Local Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/titanic-end-to-end-ml-pipeline.git](https://github.com/YOUR_USERNAME/titanic-end-to-end-ml-pipeline.git)
cd titanic-end-to-end-ml-pipeline
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```


## 📜 License
This project is open-source and available under the [MIT License](LICENSE).