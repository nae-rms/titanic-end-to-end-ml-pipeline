from data_loader import DataLoader
from preprocessor import Preprocessor
from model_trainer import ModelTrainer

def main():
    print("Data Processing Pipeline")

    print("\n1 - Loading Data ")
    loader = DataLoader('data/train.csv') # loads the data
    raw_data = loader.load_data()
    print(f"Raw Data Loaded {raw_data.shape}")

    print("\n2 - Preprocessing Data")
    preprocessor = Preprocessor(raw_data)
    preprocessor.handle_missing_values()
    preprocessor.encode_categorical_features()
    preprocessor.feature_engineering()

    final_data = preprocessor.data

    print("\n3 - Results")
    print(f"Final Data Shape: {final_data.shape}")
    print("Final Data Head")
    print(final_data.head())

    print("\n4 - Train & Test Split Data")
    trainer = ModelTrainer(final_data)
    trainer.split_data()

    print("\n5 - Training Model")
    trainer.train()

    print("\n6 - Saving Model")
    trainer.save_model()

if __name__ == "__main__":
    main()