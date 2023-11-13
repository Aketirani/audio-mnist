import unittest

import numpy as np
from sklearn.metrics import accuracy_score

from src.model_prediction import ModelPrediction


class TestModelPrediction(unittest.TestCase):
    """
    Test class for the ModelPrediction class
    """

    def setUp(self):
        """
        Set up the class with test fixtures.

        :param y_test: np.array, array for true labels
        :param y_pred: np.array, array for predicted labels
        """
        self.model_predict = ModelPrediction()
        self.y_test = np.array([0, 1, 1])
        self.y_pred = np.array([0, 1, 0])

    def test_evaluate_predictions(self):
        """
        Test the evaluate_predictions method
        """
        # calculate accuracy using the sample true labels and predicted labels
        accuracy = self.model_predict.evaluate_predictions(
            self.y_test, self.y_pred, False
        )

        # ensure accuracy is a float value
        self.assertIsInstance(accuracy, float)
