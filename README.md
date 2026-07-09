# Titanic Survival Prediction

A beginner machine learning project predicting Titanic passenger survival,
built on the classic Kaggle Titanic dataset.

## Approach
- Exploratory data analysis on Sex, Pclass, and Age patterns
- Cleaned missing Age (median per Pclass) and Embarked (mode) values
- Dropped Cabin (77% missing), PassengerId, Name, and Ticket
- Engineered an AgeGroup feature from binned Age
- One-hot encoded categorical columns (Sex, Embarked, AgeGroup)
- Trained a Logistic Regression model with an 80/20 train/validation split

## Results
- Validation accuracy: 83%
- Kaggle leaderboard score: 0.7656

## What I'd improve next
- Extract passenger title (Mr/Mrs/Miss/Master) from the Name column
- Try a Random Forest model for comparison
- Engineer a FamilySize feature from SibSp + Parch

## Files
- `titanic_prediction_model.ipynb` — full notebook, EDA through submission
- `train.csv` / `test.csv` — Kaggle's original data
- `submission.csv` — final predictions submitted to Kaggle
