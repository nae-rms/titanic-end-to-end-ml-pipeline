# End-to-End ML Prediction Pipeline

A modular and clean Machine Learning pipeline for predicting passenger survival on the Titanic dataset. Built with Python, Pandas, and Scikit-Learn, this repository provides an end-to-end framework for data loading, preprocessing, feature engineering, model training with Random Forest, and inference.

---

## 📌 Features

- **Modular Architecture**: Clean separation of concerns across dedicated modules (`DataLoader`, `Preprocessor`, `ModelTrainer`).
- **Automated Data Preprocessing**:
  - Imputation of missing values (median for `Age`, mode for `Embarked`).
  - Dropping irrelevantly detailed or high-cardinality metadata (`Cabin`, `Name`, `Ticket`, `PassengerId`).
  - Categorical encoding for `Sex` and `Embarked` using Scikit-Learn's `LabelEncoder`.
- **Feature Engineering**:
  - `FamilySize`: Combines `SibSp` and `Parch` (`SibSp + Parch + 1`).
  - `IsAlone`: Binary indicator for solo passengers (`FamilySize == 1`).
- **Machine Learning**:
  - `RandomForestClassifier` with train/test splitting and classification report metrics.
  - Automatic model persistence using `joblib`.
- **Inference Ready**: Simple script for loading pre-trained models and executing predictions on new passenger profiles.

---

## 📁 Repository Structure

```text
titanic-survival-prediction/
├── data/
│   └── train.csv                  # Input Titanic dataset
├── models/
│   └── titanic_model.pkl          # Saved trained Random Forest model
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # DataLoader class for importing dataset
│   ├── preprocessor.py            # Preprocessor class for cleaning & feature engineering
│   └── model_trainer.py           # ModelTrainer class for split, train, and save
├── main.py                        # Main script running full pipeline
├── predict.py                     # Inference script for making sample predictions
└── README.md                      # Project documentation
```

---

## 🛠️ Requirements & Installation

### Prerequisites
- Python 3.8+

### Dependencies
Install the required Python packages:

```bash
pip install pandas scikit-learn joblib
```

---

## 🚀 Usage Guide

### 1. Training the Pipeline
To execute the complete pipeline—from data ingestion through preprocessing, training, and model serialization—run `main.py`:

```bash
python main.py
```

**Pipeline Execution Flow:**
1. Loads raw dataset from `data/train.csv`.
2. Imputes missing values and removes non-predictive metadata.
3. Encodes categorical variables (`Sex`, `Embarked`).
4. Engineers `FamilySize` and `IsAlone` features.
5. Splits dataset into 80% training and 20% testing sets.
6. Trains a `RandomForestClassifier` and prints accuracy and classification metrics.
7. Saves the trained model to `models/titanic_model.pkl`.

### 2. Running Inference
To make predictions on new passenger records using the saved model:

```bash
python predict.py
```

This will load `models/titanic_model.pkl`, pass structured sample data, and output prediction results (`Survived` vs `Not Survived`).

---

## 📊 Modules & API Summary

### `src/data_loader.py` — `DataLoader`
- `DataLoader(file_path)`: Initializes loader with target dataset path.
- `load_data()`: Reads CSV file into a Pandas DataFrame.
- `get_summary()`: Prints dataset structure, summary statistics, and missing value counts.

### `src/preprocessor.py` — `Preprocessor`
- `Preprocessor(data)`: Initializes with input DataFrame.
- `handle_missing_values()`: Handles missing values for `Age` and `Embarked`, drops `Cabin`, `Name`, `Ticket`, and `PassengerId`.
- `encode_categorical_features()`: Applies label encoding to `Sex` and `Embarked`.
- `feature_engineering()`: Constructs `FamilySize` and `IsAlone`.

### `src/model_trainer.py` — `ModelTrainer`
- `ModelTrainer(data, target_column='Survived')`: Initializes trainer with preprocessed DataFrame and target variable.
- `split_data(test_size=0.2, random_state=42)`: Splits features (`X`) and targets (`y`).
- `train(model=None)`: Fits `RandomForestClassifier` (or custom estimator) and displays accuracy and classification performance.
- `save_model(file_path='models/titanic_model.pkl')`: Serializes model using `joblib`.

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
