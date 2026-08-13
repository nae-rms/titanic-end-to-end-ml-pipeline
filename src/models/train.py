import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:
    def __init__(self, data, target_column='Survived'):
        self.data = data.copy()
        self.target_column = target_column
        self.model = None

    def split_data(self, test_size=0.2, random_state=42):
        X = self.data.drop(columns = [self.target_column])
        y = self.data[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        print("Data Split Sucesfully")
        print(f"Training Set: {len(self.X_train)} samples")
        print(f"Test Set: {len(self.X_test)} samples")

    def train(self, model=None):
        if model is None:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.model = model

        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        print("Model Training Success")
        print(f"Accuracy = {accuracy:.4f}")
        print(f"Classification: {report}")

        return self

    def save_model(self, file_path='models/titanic_model.pkl'):

        if self.model is None:
            print("No model.")
            return self

        joblib.dump(self.model, file_path)
        print("Model saved.")

        return self