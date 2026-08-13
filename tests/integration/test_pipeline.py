from src.models.train import train_model

def test_train_model():
    model, X_test, y_test = train_model()

    assert model is not None
    assert X_test is not None
    assert y_test is not None