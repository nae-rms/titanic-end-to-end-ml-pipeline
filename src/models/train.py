from src.data_loader import DataLoader
from src.config import load_config
from src.features.preprocessing import create_preprocessor
from src.features.engineering import FeatureEngineer
from src.evaluation.evaluate import evaluate_model

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

def train_model():
    loader = DataLoader(
        "data/raw/train.csv"
    )

    data = loader.load_data()

    X = data.drop(columns=["Survived"])
    y = data["Survived"]

    config = load_config()
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    numerical_features = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
    categorical_features = ["Sex", "Embarked"]
    
    preprocessor = create_preprocessor(
        numerical_features,
        categorical_features
    )

    model = GradientBoostingClassifier(
        random_state=random_state
    )

    pipeline = Pipeline([
        ("feature_engineering", FeatureEngineer()),
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test

if __name__ == "__main__":
    train_model()