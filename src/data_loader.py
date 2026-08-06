import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
    
    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        print(f"Data loaded successfully! Shape: {self.data.shape}")
        return self.data

    def get_summary(self):
        if self.data is not None:
            print("Data Summary:")
            print("Column Info: \n" + str(self.data.info()))
            print("Statistical Summary: \n"+str(self.data.describe()))
            print("Missing Values: \n" + str(self.data.isnull().sum()))
        else:
            print("Data not loaded yet. Call load_data() first.")

