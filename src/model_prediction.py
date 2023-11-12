import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class ModelPrediction:
    """
    This class is used to do model predictions and evaluation
    """

    def __init__(self):
        """
        Initialize the class
        """
        pass

    @staticmethod
    def load_model(filepath: str, filename: str) -> None:
        """
        Load a pre-trained machine learning model object from a file

        :param filepath: str, the path to the directory containing the model object file
        :param filename: str, the name of the model object file
        :return: object, the loaded machine learning model
        """
        # load the model object
        try:
            return joblib.load(os.path.join(filepath, filename))
        except Exception as e:
            raise Exception(f"Error loading the model: {str(e)}")

    @staticmethod
    def predict(model_object: object, X_test: pd.DataFrame) -> pd.Series:
        """
        Make predictions on test data using a pre-loaded model

        :param model_object: object, a pre-trained machine learning model object
        :param X_test: pd.DataFrame, features for test data
        :return: pd.Series, series containing the predicted labels on test data
        """
        # make predictions on the test data and convert to pandas Series
        return pd.Series(model_object.predict(X_test))

    @staticmethod
    def evaluate_predictions(
        y_test: np.ndarray, y_pred: np.ndarray, show: bool = True
    ) -> float:
        """
        Evaluate predictions and returns model accuracy

        :param y_test: np.ndarray, true labels
        :param y_pred: np.ndarray, predicted labels
        :param show: bool, whether to print the accuracy (deault=True)
        :return: float, model accuracy
        """
        # round the predictions to the nearest integer
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)

        if show:
            print("Model Accuracy: %.2f%%" % (accuracy * 100))

        # return model accuracy
        return accuracy
