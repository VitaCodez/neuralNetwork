import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml


# Fetch the Boston Housing dataset
# The 'as_frame=True' argument loads it as a pandas DataFrame, which is convenient


'''class DataHandler:
    def __init__(self):
        boston = fetch_openml(name='boston', version=1, as_frame=True)
        X = boston.data
        y = boston.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_training_data(self):
        # Convert X_train DataFrame to a list of lists
        X_train_list = self.X_train.values.tolist()
        # Convert y_train Series to a list
        y_train_list = self.y_train.tolist()
        return self.only_floats(X_train_list), self.normalize(y_train_list)

    def get_testing_data(self):
        # Convert X_test DataFrame to a list of lists
        X_test_list = self.X_test.values.tolist()
        # Convert y_test Series to a list
        y_test_list = self.y_test.tolist()
        return self.only_floats(X_test_list), self.normalize(y_test_list)
    
    def only_floats(self, data):
        return [[math.tanh(float(x)) for x in row] for row in data]
    
    def normalize(self, data):
        # Normalize the data to the range [0, 1]

        return [x / max(data) for x in data]'''



class DataHandler:
    def __init__(self):
        boston = fetch_openml(name='boston', version=1, as_frame=True)
        X = boston.data
        y = boston.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_training_data(self):
        X_train_list = [[float(x) for x in row] for row in self.X_train.values.tolist()]
        y_train_list = [float(x) for x in self.y_train.tolist()]
        return self.normalize_features_minmax(X_train_list), self.normalize_targets_minmax(y_train_list)

    def get_testing_data(self):
        X_test_list = [[float(x) for x in row] for row in self.X_test.values.tolist()]
        y_test_list = [float(x) for x in self.y_test.tolist()]
        return self.normalize_features_minmax(X_test_list), self.normalize_targets_minmax(y_test_list)

    def normalize_features_minmax(self, data):
        # data: list of list of floats (samples x features)
        normalized = []
        # transponuj data na sloupce (feature-wise)
        data_t = list(zip(*data))

        normalized_t = []
        for col in data_t:
            col = [float(x) for x in col]  # jistota, že jsou floaty
            min_val = min(col)
            max_val = max(col)
            if max_val - min_val == 0:
                # vyhnout se dělení nulou
                normalized_col = [0 for _ in col]
            else:
                # škálování do [-1, 1]
                normalized_col = [((x - min_val) / (max_val - min_val)) * 2 - 1 for x in col]
            normalized_t.append(normalized_col)

        # transponuj zpět na řádky
        normalized = list(zip(*normalized_t))
        return [list(row) for row in normalized]

    def normalize_targets_minmax(self, data):
        min_val = min(data)
        max_val = max(data)
        if max_val - min_val == 0:
            return [0 for _ in data]
        return [((x - min_val) / (max_val - min_val)) * 2 - 1 for x in data]
