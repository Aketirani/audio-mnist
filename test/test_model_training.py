import unittest

import pandas as pd

from src.model_training import ModelTraining


class TestModelTraining(unittest.TestCase):
    """
    Test class for the ModelTraining class
    """

    def setUp(self):
        """
        Set up the class with test fixtures.

        :param model_train: class, an instance of the ModelTraining class
        :param model_param: dict, dictionary containing the parameters for the model
        :param log_data: dict, dictionary containing log data for the model
        :param expected_df: pd.DataFrame, expected result dataframe
        """
        self.model_train = ModelTraining()
        self.model_param = {
            "learning_rate": 0.05,
            "max_depth": 5,
            "n_estimators": 500,
            "gamma": 0,
            "lambda": 1,
            "scale_pos_weight": 1,
            "min_child_weight": 1,
            "objective": "binary:logistic",
            "tree_method": "hist",
        }
        self.log_data = {
            "validation_0": {"logloss": [0.5, 0.4, 0.3], "accuracy": [0.3, 0.4, 0.5]},
            "validation_1": {"logloss": [0.4, 0.3, 0.2], "accuracy": [0.2, 0.3, 0.4]},
        }
        self.expected_df = pd.DataFrame(
            {
                "iteration": [0, 1, 2],
                "train_loss": [0.5, 0.4, 0.3],
                "train_acc": [0.3, 0.4, 0.5],
                "val_loss": [0.4, 0.3, 0.2],
                "val_acc": [0.2, 0.3, 0.4],
            }
        )

    def test_set_params(self):
        """
        Test the set_params method
        """
        # call the set_params method with the sample model_param dictionary
        self.model_train.set_params(self.model_param)

        # check that the parameters of the model object have been set correctly
        self.assertEqual(
            self.model_train.model.learning_rate, self.model_param["learning_rate"]
        )
        self.assertEqual(
            self.model_train.model.max_depth, self.model_param["max_depth"]
        )
        self.assertEqual(
            self.model_train.model.n_estimators, self.model_param["n_estimators"]
        )
        self.assertEqual(
            self.model_train.model.objective, self.model_param["objective"]
        )
        self.assertEqual(
            self.model_train.model.tree_method, self.model_param["tree_method"]
        )

    def test_create_log_df(self):
        """
        Test the create_log_df method
        """
        # get the resulting DataFrame from the method
        result_df = self.model_train.create_log_df(self.log_data)

        # check if the resulting DataFrame match the expected one
        pd.testing.assert_frame_equal(result_df, self.expected_df)
