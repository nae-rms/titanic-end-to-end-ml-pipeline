from src.data_loader import DataLoader
from sklearn.model_selection import train_test_split
from src.config import load_config
from src.features.preprocessing import create_preprocessor
from sklearn.pipeline import Pipeline
from src.features.engineering import FeatureEngineer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def test_load_titanic_data():
    loader = DataLoader(
        "/Users/wallace/Documents/Projects/titanic-end-to-end-model/data/raw/train.csv"
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
        random_state=random_state
    )

    numerical_features = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
    categorical_features = ["Sex", "Embarked"]

    preprocessor = create_preprocessor(
        numerical_features,
        categorical_features
    )

    model = LogisticRegression()

    pipeline = Pipeline([
        ("feature_engineering", FeatureEngineer()),
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)  
    predictions = pipeline.predict(X_test)
    print(predictions[:10])

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    
