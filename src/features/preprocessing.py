from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def create_preprocessor(numerical_features, categorical_features):
    return ColumnTransformer([
        ("numeric", numeric_pipeline(), numerical_features),
        ("categorical", categorical_pipeline(), categorical_features)
    ])

def numeric_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

def categorical_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder())
    ])

