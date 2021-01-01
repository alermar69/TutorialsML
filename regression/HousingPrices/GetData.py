import os
import pandas as pd

DATA_PATH = os.path.join("datasets", "HousingPrices")

def load_data(path=DATA_PATH):
    csv_path = os.path.join(path, 'train.csv')
    return pd.read_csv(csv_path)
