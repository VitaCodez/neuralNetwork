import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml


# Fetch the Boston Housing dataset
# The 'as_frame=True' argument loads it as a pandas DataFrame, which is convenient


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
        return [[float(x) for x in row] for row in data]
    
    def normalize(self, data):
        # Normalize the data to the range [0, 1]

        return [x / max(data) for x in data]
