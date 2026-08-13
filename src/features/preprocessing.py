import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self, data):
        self.data = data.copy()

    def handle_missing_values(self):
        self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())
        self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode()[0])
        self.data = self.data.drop(columns=['Cabin'], errors='ignore')
        self.data = self.data.drop(columns=['Name'], errors='ignore')
        self.data = self.data.drop(columns=['Ticket'], errors='ignore')
        self.data = self.data.drop(columns=['PassengerId'], errors='ignore')

        print("Missing values handled.")
        return self

    def encode_categorical_features(self):
        label_encoders = {}

        for col in ['Sex', 'Embarked']:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col].astype(str))
            label_encoders[col] = le

        print("Categorical features encoded.")
        self.label_encoders = label_encoders
        return self

    def feature_engineering(self):
        self.data['FamilySize'] = self.data['SibSp'] + self.data['Parch'] + 1
        self.data['IsAlone'] = (self.data['FamilySize'] == 1).astype(int)

        print("New features created.")
        return self